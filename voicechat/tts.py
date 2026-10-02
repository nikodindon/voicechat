"""Synthèse vocale Kokoro, locale, avec accélérateur GPU si présent.

Deux niveaux :
  * ``KokoroTTS``     — charge un pipeline Kokoro et rend un tableau numpy ;
  * ``SpeechPipeline``— thread + file : on lui donne du texte, il synthétise
                        et joue, sans bloquer la lecture du flux LLM.
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .audio import Speaker

SAMPLE_RATE = 24000  # Kokoro-82M génère en 24 kHz mono
REPO_ID = "hexgrad/Kokoro-82M"  # passé explicitement : sinon Kokoro logue un avertissement


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
        self._pipeline.load_voice(voice)
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
        for _, _, audio in self._pipeline(text, voice=self.voice, speed=self.speed):
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
        self.enabled = True
        self.synth_errors = 0

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

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            try:
                if item is None:
                    return
                phrase = self.cleaner(item)
                if not phrase:
                    continue
                try:
                    audio = self.tts.synth(phrase)
                except Exception:
                    self.synth_errors += 1
                    continue
                self.speaker.play(audio)
            finally:
                self._q.task_done()
