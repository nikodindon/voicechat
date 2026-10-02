"""Lecture audio : un flux de sortie continu, alimenté par une file de tampons.

Pourquoi un flux persistant ? ``sounddevice.play()`` ouvre puis referme un flux à chaque
appel : sur ce matériel c'est ~10-30 ms de silence entre deux phrases — précisément le
« trou » que l'on cherche à supprimer. Ici le flux reste ouvert pour toute la session et
un rappel (callback) tire les échantillons au fil de l'eau : la lecture est continue, et
il n'y a plus de latence de mise en route à chaque phrase.

Corollaire utile : on sait exactement combien d'échantillons sont sortis, donc « attendre
la fin de la voix » ne repose plus sur une approximation.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

try:  # sounddevice est optionnel : sans lui, le mode --no-tts reste utilisable
    import sounddevice as sd

    _SD_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - dépend du système
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERROR = str(exc)

BLOCSIZE = 1024  # échantillons par appel du rappel (~43 ms à 24 kHz)


class Speaker:
    """Flux de sortie persistant : on empile des tampons, le rappel les joue."""

    def __init__(self, samplerate: int = 24000, device: int | str | None = None) -> None:
        self.samplerate = samplerate
        self.device = device
        self._blocs: deque = deque()
        self._courant: np.ndarray | None = None
        self._pos = 0
        self._verrou = threading.Lock()
        self._flux = None
        self._frames_pousses = 0
        self._frames_joues = 0
        self._condition = threading.Condition()
        self.last_error: str | None = None

    # ---------------------------------------------------------------- cycle de vie
    @property
    def available(self) -> bool:
        """La bibliothèque audio est-elle présente ?"""
        return sd is not None

    @property
    def pret(self) -> bool:
        """Le flux de sortie est-il ouvert et utilisable ?"""
        return self._flux is not None

    def start(self) -> None:
        if sd is None or self._flux is not None:
            return
        try:
            self._flux = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                blocksize=BLOCSIZE,
                device=self.device,  # type: ignore[arg-type]
                callback=self._rappel,
            )
            self._flux.start()
        except Exception as exc:
            self.last_error = f"sortie audio indisponible : {exc}"
            self._flux = None

    def play(self, audio) -> None:
        """Empile un tampon (retour immédiat : c'est le rappel qui le joue)."""
        if self._flux is None or audio is None:
            return
        bloc = np.asarray(audio, dtype=np.float32).reshape(-1)
        if bloc.size == 0:
            return
        with self._verrou:
            self._blocs.append(bloc)
            self._frames_pousses += bloc.size
        with self._condition:
            self._condition.notify_all()

    def wait(self) -> None:
        """Attend que tout ce qui a été empilé soit réellement sorti."""
        if self._flux is None:
            return
        with self._condition:
            while self._frames_joues < self._frames_pousses:
                self._condition.wait(timeout=0.1)
        # Le rappel a livré les échantillons, mais le matériel en garde encore en tampon.
        try:
            latence = float(self._flux.latency)
        except Exception:  # pragma: no cover
            latence = 0.0
        if latence > 0:
            time.sleep(latence)

    def flush(self) -> None:
        """Coupe net : jette la file **et** ce que le matériel a déjà en tampon.

        ``stop()`` laisserait finir ce qui est en mémoire ; ``abort()`` le jette, ce qui
        est exactement ce qu'on veut pour un Ctrl+C.
        """
        with self._verrou:
            self._blocs.clear()
            self._courant = None
            self._pos = 0
            self._frames_pousses = self._frames_joues
        if self._flux is None:
            return
        try:
            self._flux.abort()
            self._flux.start()  # abort() a terminé le flux : il faut le relancer
        except Exception as exc:
            self.last_error = f"coupure audio : {exc}"

    def close(self) -> None:
        self.flush()
        if self._flux is not None:
            try:
                self._flux.stop()
                self._flux.close()
            except Exception:  # pragma: no cover
                pass
            self._flux = None

    @property
    def played_s(self) -> float:
        """Durée d'audio réellement sortie sur le périphérique."""
        return self._frames_joues / self.samplerate

    # ------------------------------------------------------------------- interne
    def _rappel(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        """Appelé par PortAudio : remplit ``outdata`` depuis la file."""
        ecrit = 0
        with self._verrou:
            while ecrit < frames:
                if self._courant is None:
                    if not self._blocs:
                        break
                    self._courant = self._blocs.popleft()
                    self._pos = 0
                prendre = min(self._courant.size - self._pos, frames - ecrit)
                # outdata est de forme (frames, canaux) ; on ouvre toujours en mono.
                outdata[ecrit : ecrit + prendre, 0] = self._courant[self._pos : self._pos + prendre]
                self._pos += prendre
                ecrit += prendre
                if self._pos >= self._courant.size:
                    self._courant = None
            if ecrit < frames:
                outdata[ecrit:, 0] = 0.0  # silence si la voix est en retard
            self._frames_joues += ecrit
        if ecrit:
            with self._condition:
                self._condition.notify_all()

    # ------------------------------------------------------------------- debug
    def describe_devices(self) -> str:
        if sd is None:
            return f"sounddevice indisponible : {_SD_IMPORT_ERROR}"
        lignes = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                lignes.append(f"  [{idx}] {dev['name']} ({int(dev['default_samplerate'])} Hz)")
        return "\n".join(lignes) or "  (aucune sortie audio détectée)"
