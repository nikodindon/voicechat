"""Synthèse vocale Kokoro, locale, avec accélérateur GPU si présent.

Deux niveaux :
  * ``KokoroTTS``     — charge un pipeline Kokoro et rend un tableau numpy ;
  * ``SpeechPipeline``— thread + file : on lui donne du texte, il synthétise
                        et joue, sans bloquer la lecture du flux LLM.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import re
import sys
import threading
import time

import numpy as np

from .audio import Speaker

SAMPLE_RATE = 24000  # Kokoro-82M génère en 24 kHz mono
REPO_ID = "hexgrad/Kokoro-82M"  # passé explicitement : sinon Kokoro logue un avertissement

# Trace de ce qui part réellement à la synthèse. Utile pour vérifier le nettoyage
# du markdown de bout en bout (on voit le texte après découpage ET nettoyage),
# là où lire la réponse à l'écran ne prouve rien.
TRACE_TTS = os.environ.get("VOICECHAT_TRACE_TTS") == "1"


# ------------------------------------------------------------------ utilitaires
def _cuda_smoke_test(torch) -> tuple[bool, str]:
    """Vérifie par un vrai calcul que le GPU est utilisable, pas seulement visible.

    `torch.cuda.is_available()` ment dans un cas courant : le wheel connaît le GPU
    mais ne contient plus ses noyaux (Pascal sm_61 retiré des builds CUDA >= 12.8).
    On ne le découvre qu'à la première opération — donc ici, volontairement.
    """
    try:
        sonde = torch.zeros(8, device="cuda")
        (sonde + 1).sum().item()
        torch.cuda.synchronize()
        del sonde
        return True, ""
    except Exception as exc:
        court = str(exc).strip().splitlines()[0][:120]
        try:
            archs = ", ".join(torch.cuda.get_arch_list())
        except Exception:  # pragma: no cover
            archs = "?"
        cc = torch.cuda.get_device_capability(0)
        return False, (
            f"GPU visible (sm_{cc[0]}{cc[1]}) mais aucun noyau compilé pour lui — "
            f"ce wheel ne contient que {archs}. Détail : {court}"
        )


def probe_cuda() -> tuple[bool, str]:
    """(cuda_vraiment_utilisable, description lisible) — ne lève jamais."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return False, f"torch absent ({exc})"

    if not torch.cuda.is_available():
        version_cuda = getattr(torch.version, "cuda", None)
        if version_cuda is None:
            return False, f"torch {torch.__version__} sans support CUDA"
        return False, f"torch {torch.__version__}, CUDA {version_cuda} présent mais aucun GPU visible"

    ok, probleme = _cuda_smoke_test(torch)
    if not ok:
        return False, f"torch {torch.__version__} — {probleme}"

    try:
        nom = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:  # pragma: no cover
        nom, mem = "GPU", 0
    return True, f"torch {torch.__version__} — {nom} ({mem} Mo)"


def resolve_device(pref: str) -> tuple[str, str]:
    """Traduit 'auto'|'cuda'|'cpu' en device torch, avec la raison du choix."""
    dispo, description = probe_cuda()
    pref = (pref or "auto").lower()

    if pref == "cpu":
        return "cpu", "CPU demandé explicitement"
    if pref == "cuda":
        if dispo:
            return "cuda", f"CUDA demandé — {description}"
        return "cpu", f"CUDA demandé MAIS inutilisable → repli CPU. {description}"
    if dispo:
        return "cuda", f"auto → GPU. {description}"
    return "cpu", f"auto → CPU. {description}"



def list_voices() -> list[str]:
    """Liste les voix Kokoro du dépôt Hugging Face (mise en cache locale)."""
    try:
        from huggingface_hub import list_repo_files
    except Exception:
        return []
    try:
        fichiers = list_repo_files(REPO_ID, repo_type="model")
    except Exception:
        return []
    return sorted(
        f[len("voices/") : -len(".pt")]
        for f in fichiers
        if f.startswith("voices/") and f.endswith(".pt")
    )


# ------------------------------------------------------------------ voix

@contextlib.contextmanager
def _sans_avertissement_langue():
    """Tait le « Language mismatch » de Kokoro le temps d'un chargement.

    Kokoro avertit quand on charge une voix d'une autre langue que celle du pipeline.
    Dans un mélange de voix, c'est justement ce qu'on demande : le message est du
    bruit. On filtre ce message-là et aucun autre.
    """
    class _Filtre(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                return not record.getMessage().startswith("Language mismatch")
            except Exception:
                return True

    filtre = _Filtre()
    loggers = [
        logging.getLogger(nom)
        for nom in ("kokoro", "kokoro.pipeline", "kokoro.model", "kokoro.istftnet")
    ]
    for logger in loggers:
        logger.addFilter(filtre)
    try:
        yield
    finally:
        for logger in loggers:
            logger.removeFilter(filtre)


def analyser_voix(spec: str) -> list[tuple[str, float]]:
    """« ff_siwis:3+ef_dora:1 » → ``[('ff_siwis', 0.75), ('ef_dora', 0.25)]``.

    Les poids sont normalisés pour sommer à 1 : c'est l'échelle du mélange natif de
    Kokoro, qui fait la moyenne des styles. Une voix seule garde donc exactement le
    rendu d'avant, et `ff_siwis,ef_dora` (la syntaxe Kokoro) donne bien 50/50.
    """
    parts: list[tuple[str, float]] = []
    for morceau in re.split(r"[+,]", spec):
        morceau = morceau.strip()
        if not morceau:
            continue
        nom, _, poids_txt = morceau.partition(":")
        nom = nom.strip()
        if not nom:
            raise ValueError(f"nom de voix vide dans « {spec} »")
        poids = 1.0
        if poids_txt.strip():
            try:
                poids = float(poids_txt)
            except ValueError:
                raise ValueError(
                    f"poids illisible « {poids_txt.strip()} » dans « {spec} »"
                ) from None
            if poids <= 0:
                raise ValueError(f"poids nul ou négatif pour « {nom} »")
        parts.append((nom, poids))
    if not parts:
        raise ValueError(f"aucune voix dans « {spec} »")
    total = sum(p for _, p in parts)
    return [(nom, p / total) for nom, p in parts]


def est_melange(spec: str) -> bool:
    """Vrai si la spécification demande plusieurs voix (donc un mélange à composer)."""
    return len(analyser_voix(spec)) > 1


def decrire_voix(spec: str) -> str:
    """Résumé lisible : ``ff_siwis 80 % + ef_dora 20 %``.

    Ne lève jamais : sur une spécification illisible on rend le texte brut, pour que
    l'affichage d'une erreur ne provoque pas une seconde erreur.
    """
    try:
        parts = analyser_voix(spec)
    except ValueError:
        return spec
    if len(parts) == 1:
        return parts[0][0]
    return " + ".join(f"{nom} {poids:.0%}" for nom, poids in parts)


# ------------------------------------------------------------------ synthétiseur
class KokoroTTS:
    """Charge une voix Kokoro et la garde en mémoire entre deux phrases."""

    def __init__(
        self,
        voice: str = "ff_siwis",
        lang_code: str = "f",
        speed: float = 1.0,
        device: str = "auto",
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.speed = float(speed)
        self.device_pref = device
        self.device = "cpu"
        self.device_reason = ""
        self.load_error: str | None = None
        self._pipeline = None
        self._loaded_voice: str | None = None
        # Ce qu'on passe réellement au pipeline : un nom de voix, ou un tenseur de
        # style quand c'est un mélange pondéré (Kokoro ne sait pas pondérer).
        self._voix_chargee = None
        self.synth_s = 0.0  # temps cumulé passé à synthétiser
        self.audio_s = 0.0  # durée cumulée d'audio produit

    # ------------------------------------------------------------------ chargement
    def load(self) -> bool:
        """Charge Kokoro. Retourne False (sans lever) si le TTS est inutilisable."""
        try:
            import torch
            from kokoro import KPipeline
        except Exception as exc:
            self.load_error = f"Kokoro/torch non installé : {exc}"
            return False

        # resolve_device() fait déjà le test de fumée CUDA : si le GPU n'est que
        # « visible » sans noyaux compilés, on arrive ici directement en CPU.
        self.device, self.device_reason = resolve_device(self.device_pref)


        # Selon la version de kokoro, `device` est accepté ou non par KPipeline.
        try:
            import inspect

            params = inspect.signature(KPipeline.__init__).parameters
            kwargs = {"repo_id": REPO_ID}
            if "device" in params:
                kwargs["device"] = self.device
            elif self.device == "cpu":
                # Repli : forcer la détection interne de torch vers le CPU.
                torch.cuda.is_available = lambda: False  # type: ignore[assignment]
                self.device_reason += " (forcé via torch.cuda.is_available)"
            self._pipeline = KPipeline(lang_code=self.lang_code, **kwargs)
        except TypeError as exc:
            self.load_error = f"KPipeline inutilisable avec ce jeu de paramètres : {exc}"
            return False

        # Le pipeline expose `device` ; on s'aligne sur ce qu'il a réellement choisi.
        self.device = getattr(self._pipeline, "device", self.device)
        try:
            self._load_voice(self.voice)
        except Exception as exc:
            self.load_error = f"voix « {self.voice} » introuvable : {exc}"
            return False
        return True

    def _load_voice(self, voice: str) -> None:
        """Charge la voix, ou compose le mélange pondéré.

        Kokoro sait moyenner plusieurs voix (`ff_siwis,ef_dora`) mais pas les pondérer.
        Il accepte en revanche un **tenseur de style** : on compose donc le mélange
        nous-mêmes et on le lui passe tel quel. Une voix seule suit le chemin normal,
        donc son rendu ne change pas d'un iota.
        """
        parts = analyser_voix(voice)
        if len(parts) == 1:
            self._pipeline.load_voice(parts[0][0])
            self._voix_chargee = parts[0][0]
        else:
            # Les voix d'autres langues déclenchent un « Language mismatch » — c'est
            # précisément ce qu'on demande dans un mélange, donc on le fait taire.
            with _sans_avertissement_langue():
                packs = [
                    (self._pipeline.load_single_voice(nom), poids)
                    for nom, poids in parts
                ]
            melange = packs[0][0] * packs[0][1]
            for pack, poids in packs[1:]:
                melange = melange + pack * poids
            self._voix_chargee = melange
        self._loaded_voice = voice

    def set_voice(self, voice: str) -> None:
        self._load_voice(voice)
        self.voice = voice

    def set_lang(self, lang_code: str) -> None:
        """Change la langue : reconstruit le pipeline (G2P différent)."""
        from kokoro import KPipeline

        self.lang_code = lang_code
        self._pipeline = KPipeline(lang_code=lang_code, device=self.device, repo_id=REPO_ID)
        self._loaded_voice = None
        self._load_voice(self.voice)

    @property
    def ready(self) -> bool:
        return self._pipeline is not None and self._loaded_voice is not None

    # ------------------------------------------------------------------ synthèse
    def synth(self, text: str) -> np.ndarray:
        """Texte → tableau float32 mono 24 kHz."""
        if not self.ready:
            raise RuntimeError(self.load_error or "TTS non chargé")
        t0 = time.monotonic()
        morceaux: list[np.ndarray] = []
        for _, _, audio in self._pipeline(text, voice=self._voix_chargee, speed=self.speed):
            if audio is None:
                continue
            if hasattr(audio, "detach"):  # tenseur torch
                audio = audio.detach().to("cpu").numpy()
            morceaux.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not morceaux:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(morceaux)
        self.synth_s += time.monotonic() - t0
        self.audio_s += len(out) / SAMPLE_RATE
        return out

    @property
    def rtf(self) -> float:
        """Real-Time Factor : < 1 signifie « synthétise plus vite que ça ne se joue »."""
        return self.synth_s / self.audio_s if self.audio_s > 0 else 0.0


# --------------------------------------------------------------- file + lecture
class SpeechPipeline:
    """File de phrases → synthèse → lecture, dans un thread unique.

    On empile du TEXTE ; le thread synthétise puis joue. Un seul thread donc
    l'ordre est garanti et le GPU n'est jamais sollicité deux fois en même temps.
    """

    def __init__(self, tts: KokoroTTS, speaker: Speaker, cleaner=None) -> None:
        self.tts = tts
        self.speaker = speaker
        self.cleaner = cleaner or (lambda t: t)
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Incrémenté à chaque vidage : une synthèse commencée avant le vidage et
        # terminée après ne doit pas être jouée (sinon une phrase « fantôme »
        # sort après un Ctrl+C).
        self._epoque = 0
        self.enabled = True
        self.synth_errors = 0
        # Phrases dont la synthèse a échoué (GPU en vrac, mémoire pleine) : on les
        # garde au lieu de les jeter, pour pouvoir les réentendre une fois le GPU
        # revenu. Borné, parce qu'une longue session ne doit pas remplir la mémoire.
        self._en_attente: list[str] = []
        self.max_attente = 40

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        """Empile une phrase (non bloquant)."""
        if not self.enabled or not text.strip():
            return
        self._q.put(text)

    def wait(self) -> None:
        """Attend que la file soit vide ET que le son en cours soit fini."""
        self._q.join()
        self.speaker.wait()

    def flush(self) -> None:
        """Coupe la voix : vide la file, arrête le son, et invalide la synthèse en cours."""
        self._epoque += 1
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass
        self.speaker.flush()

    def close(self) -> None:
        self._stop.set()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def pending(self) -> int:
        return self._q.qsize()

    # ------------------------------------------------------------- mode dégradé
    def _retenir(self, phrase: str) -> None:
        """Met une phrase de côté après un échec de synthèse.

        On garde les plus **récentes** : c'est la fin de la réponse qui vient d'être
        lue à l'écran, donc celle dont l'utilisateur se souvient.
        """
        self._en_attente.append(phrase)
        if len(self._en_attente) > self.max_attente:
            self._en_attente.pop(0)

    @property
    def en_attente(self) -> int:
        """Nombre de phrases dont la synthèse a échoué et qui attendent un rejeu."""
        return len(self._en_attente)

    @property
    def en_panne(self) -> bool:
        """Vrai si des phrases ont été perdues pour la voix (le texte, lui, est intact)."""
        return bool(self._en_attente)

    def rejouer(self) -> int:
        """Remet en file les phrases non synthétisées. Renvoie combien.

        Le pipeline ne retente **pas** tout seul : rejouer en plein milieu d'une
        réponse mélangerait deux textes. C'est l'utilisateur qui décide du moment,
        quand il sait que le GPU est revenu.
        """
        phrases, self._en_attente = self._en_attente, []
        for phrase in phrases:
            self.say(phrase)
        return len(phrases)

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            try:
                if item is None:
                    return
                epoque = self._epoque
                phrase = self.cleaner(item)
                if not phrase:
                    continue
                if TRACE_TTS:
                    print(f"[tts] {phrase}", file=sys.stderr, flush=True)
                try:
                    audio = self.tts.synth(phrase)
                except Exception:
                    self.synth_errors += 1
                    self._retenir(phrase)
                    continue
                if epoque != self._epoque:
                    # Un vidage (Ctrl+C) est passé pendant la synthèse : on jette,
                    # sinon la phrase sortirait après l'interruption.
                    continue
                self.speaker.play(audio)
            finally:
                self._q.task_done()
