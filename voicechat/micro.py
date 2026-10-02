"""Écoute du microphone : capture 16 kHz, détection de parole, découpage en énoncés.

Le VAD utilisé est **Silero v6**, celui que `faster-whisper` embarque déjà
(``faster_whisper/assets/silero_vad_v6.onnx``) : aucune dépendance supplémentaire,
`onnxruntime` étant déjà là. Son intérêt sur un simple seuil d'énergie est décisif :
mesuré sur cette machine, il donne 0,02 sur du silence, 0,06 sur du bruit fort et
0,74 en moyenne sur de la vraie parole. Un seuil d'énergie, lui, se déclenche sur
n'importe quel bruit de fond — et ici le micro est bruyant.

L'interface du modèle v6 n'est pas celle de v5 : l'entrée fait **576** échantillons
(64 de contexte + 512 nouveaux) et les états LSTM sont deux tenseurs séparés ``h``/``c``.
"""

from __future__ import annotations

import os
import queue
from collections import deque

import numpy as np

try:  # sounddevice est optionnel, comme pour la sortie
    import sounddevice as sd

    _SD_ERREUR: str | None = None
except Exception as exc:  # pragma: no cover - dépend du système
    sd = None  # type: ignore[assignment]
    _SD_ERREUR = str(exc)

FREQUENCE = 16000  # le VAD et Whisper travaillent tous deux en 16 kHz


def chemin_vad() -> str:
    """Chemin du modèle VAD livré avec faster-whisper."""
    import faster_whisper

    base = os.path.dirname(faster_whisper.__file__)
    for nom in ("silero_vad_v6.onnx", "silero_vad.onnx"):
        chemin = os.path.join(base, "assets", nom)
        if os.path.isfile(chemin):
            return chemin
    raise FileNotFoundError(
        "modèle VAD introuvable dans faster_whisper/assets — "
        "faster-whisper est-il bien installé ?"
    )


class DetecteurParole:
    """Dit, toutes les 32 ms, si le bloc contient de la parole."""

    TAILLE_BLOC = 512  # 32 ms à 16 kHz : contrainte du modèle
    CONTEXTE = 64  # échantillons du bloc précédent à recoller en tête

    def __init__(self, seuil: float = 0.5, chemin: str | None = None) -> None:
        self.seuil = seuil
        self._session = None
        self.erreur = ""
        try:
            import onnxruntime

            # Un seul fil d'exécution : le VAD est appelé toutes les 32 ms, il n'a rien
            # à gagner au parallélisme, et des fils en plus ne feraient que disputer le
            # GIL au rappel audio de PortAudio — ce qui faisait perdre des échantillons.
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = onnxruntime.InferenceSession(
                chemin or chemin_vad(),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            self.erreur = f"VAD indisponible : {exc}"
        self.reinitialiser()

    def reinitialiser(self) -> None:
        """Remet les états LSTM à zéro — à faire au début de chaque écoute."""
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._contexte = np.zeros(self.CONTEXTE, dtype=np.float32)

    @property
    def pret(self) -> bool:
        return self._session is not None

    def probabilite(self, bloc: np.ndarray) -> float:
        if self._session is None:
            return 0.0
        entree = np.concatenate([self._contexte, bloc]).reshape(1, -1).astype(np.float32)
        probs, self._h, self._c = self._session.run(
            None, {"input": entree, "h": self._h, "c": self._c}
        )
        self._contexte = bloc[-self.CONTEXTE :].copy()
        return float(np.asarray(probs).ravel()[0])

    def contient_parole(self, bloc: np.ndarray) -> bool:
        return self.probabilite(bloc) >= self.seuil


class Microphone:
    """Capture 16 kHz mono et débite des blocs de taille fixe.

    Le découpage en blocs est refait ici plutôt que demandé au matériel : toutes les
    cartes n'acceptent pas 512 échantillons par appel, et le VAD exige exactement ça.
    """

    def __init__(
        self,
        taille_bloc: int = DetecteurParole.TAILLE_BLOC,
        device: int | str | None = None,
        frequence: int = FREQUENCE,
        prechauffe: float = 0.15,
    ) -> None:
        self.taille_bloc = taille_bloc
        self.device = device
        self.frequence = frequence
        # À l'ouverture, beaucoup de cartes livrent un ou deux blocs d'artefact
        # (claquement, valeurs saturées). Les jeter évite de faire croire au VAD
        # qu'une phrase commence, et évite de polluer le début d'une transcription.
        self.prechauffe = prechauffe
        self._a_jeter = 0
        self._file: queue.Queue = queue.Queue()
        self._reste = np.zeros(0, dtype=np.float32)
        self._flux = None
        self.erreur: str | None = None
        # Débordements signalés par PortAudio : chaque débordement = des échantillons
        # perdus, donc un mot haché. On les compte pour pouvoir le dire.
        self.pertes = 0

    @property
    def available(self) -> bool:
        return sd is not None

    @property
    def pret(self) -> bool:
        return self._flux is not None

    def ouvrir(self) -> bool:
        if sd is None:
            self.erreur = f"sounddevice indisponible ({_SD_ERREUR})"
            return False
        try:
            self._flux = sd.InputStream(
                samplerate=self.frequence,
                channels=1,
                dtype="float32",
                # Gros blocs + latence haute : PortAudio garde un tampon plus large,
                # donc un rappel retardé par le GIL (le VAD prend la main) ne fait
                # plus perdre d'échantillons. Sans ça on perdait 128 ms sur 8 s, et
                # les mots hachés se retrouvaient mal transcrits.
                blocksize=self.taille_bloc * 8,
                latency="high",
                device=self.device,  # type: ignore[arg-type]
                callback=self._rappel,
            )
            self._flux.start()
        except Exception as exc:
            self.erreur = f"micro indisponible : {exc}"
            self._flux = None
            return False
        self._a_jeter = max(0, int(self.prechauffe * self.frequence / self.taille_bloc))
        return True

    def fermer(self) -> None:
        if self._flux is not None:
            try:
                self._flux.stop()
                self._flux.close()
            except Exception:  # pragma: no cover
                pass
            self._flux = None
        with self._file.mutex:
            self._file.queue.clear()
        self._reste = np.zeros(0, dtype=np.float32)

    def __enter__(self) -> "Microphone":
        self.ouvrir()
        return self

    def __exit__(self, *exc) -> bool:
        self.fermer()
        return False

    def _rappel(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status and status.input_overflow:
            self.pertes += 1
        self._file.put(indata[:, 0].copy())

    def lire(self, delai: float = 0.5) -> np.ndarray | None:
        """Le prochain bloc, ou ``None`` si rien n'arrive dans le délai imparti."""
        while True:
            while self._reste.size < self.taille_bloc:
                try:
                    morceau = self._file.get(timeout=delai)
                except queue.Empty:
                    return None
                self._reste = np.concatenate([self._reste, morceau])
            bloc = self._reste[: self.taille_bloc]
            self._reste = self._reste[self.taille_bloc :]
            if self._a_jeter > 0:
                self._a_jeter -= 1
                continue  # bloc d'échauffement : jeté, mais l'alignement est conservé
            return bloc


class Ecouteur:
    """Découpe le flux du micro en énoncés complets.

    Un énoncé commence au premier bloc jugé « parole » (en gardant un peu d'audio
    d'avant, sinon on coupe la première syllabe) et se termine après ``silence_fin``
    secondes de silence. Les énoncés plus courts que ``duree_min`` sont jetés : c'est
    ce qui élimine les toussotements et les clics.
    """

    def __init__(
        self,
        micro: Microphone | None = None,
        vad: DetecteurParole | None = None,
        *,
        seuil: float = 0.5,
        silence_fin: float = 0.7,
        duree_min: float = 0.35,
        duree_max: float = 30.0,
        pre_roll: float = 0.25,
        device: int | str | None = None,
        on_parole=None,
    ) -> None:
        self.vad = vad or DetecteurParole(seuil=seuil)
        self.micro = micro or Microphone(device=device)
        self.silence_fin = silence_fin
        self.duree_min = duree_min
        self.duree_max = duree_max
        self.pre_roll = pre_roll
        self.on_parole = on_parole  # appelé quand on détecte le début de la parole

    @property
    def pret(self) -> bool:
        return self.micro.available

    # ------------------------------------------------------------------ écoute
    def ecouter(self, delai_depart: float = 60.0) -> np.ndarray | None:
        """Attend un énoncé et le retourne en float32 16 kHz.

        ``None`` si rien n'a été dit dans ``delai_depart`` secondes, ou si le micro
        ne répond plus. Une interruption clavier (Ctrl+C) remonte telle quelle.
        """
        bloc_s = self.vad.TAILLE_BLOC / FREQUENCE
        # round() plutôt que int() : la division flottante d'une durée par la taille
        # d'un bloc peut tomber juste sous l'entier (2,999…), et on perdrait un bloc.
        pre_roll = deque(maxlen=max(1, round(self.pre_roll / bloc_s)))
        morceaux: list[np.ndarray] = []
        en_parole = False
        silence = 0.0
        attente = 0.0
        self.vad.reinitialiser()

        with self.micro:
            if not self.micro.pret:
                return None
            while True:
                bloc = self.micro.lire()
                if bloc is None:
                    if en_parole:
                        break  # le micro s'est tu en pleine phrase : on termine là
                    attente += 0.5
                    if attente >= delai_depart:
                        return None
                    continue

                parle = self.vad.contient_parole(bloc)

                if not en_parole:
                    if parle:
                        # Le bloc déclencheur est ajouté à la suite du pré-roll, sans
                        # passer par la deque : sinon il évince le plus ancien et le
                        # pré-roll fait toujours un bloc de moins que demandé.
                        en_parole = True
                        morceaux = [*pre_roll, bloc]
                        silence = 0.0
                        if self.on_parole:
                            self.on_parole()
                    else:
                        pre_roll.append(bloc)
                    continue

                morceaux.append(bloc)
                silence = 0.0 if parle else silence + bloc_s
                if silence >= self.silence_fin:
                    break
                if sum(m.size for m in morceaux) / FREQUENCE >= self.duree_max:
                    break

        if not morceaux:
            return None
        audio = np.concatenate(morceaux).astype(np.float32)
        if audio.size / FREQUENCE < self.duree_min:
            return None  # trop court pour être une phrase
        return audio
