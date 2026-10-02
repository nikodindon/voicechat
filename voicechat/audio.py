"""Lecture audio : une file d'attente consommée par un thread dédié.

Pourquoi un thread ? Parce que le son doit continuer à sortir pendant qu'on
lit la suite du flux LLM. Le thread de lecture est le seul à toucher à la
carte son, ce qui évite les accès concurrents à PortAudio.
"""

from __future__ import annotations

import queue
import threading
import time

try:  # sounddevice est optionnel : sans lui, le mode --no-tts reste utilisable
    import sounddevice as sd

    _SD_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - dépend du système
    sd = None  # type: ignore[assignment]
    _SD_IMPORT_ERROR = str(exc)


class Speaker:
    """File de tampons audio joués dans l'ordre, un par un."""

    def __init__(self, samplerate: int = 24000, device: int | str | None = None) -> None:
        self.samplerate = samplerate
        self.device = device
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._current: object | None = None
        self._lock = threading.Lock()
        self.played_s = 0.0
        self.last_error: str | None = None

    # ---------------------------------------------------------------- cycle de vie
    @property
    def available(self) -> bool:
        return sd is not None

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
        self._thread.start()

    def play(self, audio) -> None:
        """Met un tampon audio en file (retour immédiat)."""
        if not self.available or audio is None or len(audio) == 0:
            return
        self._q.put(audio)

    def wait(self) -> None:
        """Bloque jusqu'à ce que tout ce qui est en file ait été joué."""
        self._q.join()

    def flush(self) -> None:
        """Vide la file et coupe le son en cours (Ctrl+C / nouvelle question)."""
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass
        if sd is not None:
            try:
                sd.stop()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        self._stop.set()
        self.flush()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------- interne
    def _run(self) -> None:
        assert sd is not None
        while not self._stop.is_set():
            item = self._q.get()
            try:
                if item is None:
                    return
                with self._lock:
                    self._current = item
                duree = float(len(item)) / self.samplerate
                try:
                    sd.play(item, samplerate=self.samplerate, device=self.device)
                    sd.wait()
                    self.played_s += duree
                except Exception as exc:  # carte son absente, périphérique occupé…
                    self.last_error = f"lecture audio impossible : {exc}"
                finally:
                    with self._lock:
                        self._current = None
            finally:
                self._q.task_done()
            time.sleep(0)  # laisse respirer le GIL entre deux phrases

    # ------------------------------------------------------------------- debug
    def describe_devices(self) -> str:
        if sd is None:
            return f"sounddevice indisponible : {_SD_IMPORT_ERROR}"
        lignes = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_output_channels", 0) > 0:
                lignes.append(f"  [{idx}] {dev['name']} ({int(dev['default_samplerate'])} Hz)")
        return "\n".join(lignes) or "  (aucune sortie audio détectée)"
