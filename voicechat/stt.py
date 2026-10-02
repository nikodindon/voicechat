"""Reconnaissance vocale locale, via faster-whisper (CTranslate2).

Piège propre à cette machine : **Pascal ne supporte pas le float16** en calcul
efficace, et CTranslate2 refuse carrément de démarrer avec ::

    ValueError: Requested float16 compute type, but the target device or backend
    do not support efficient float16 computation.

Le choix « évident » (``compute_type="float16"``) est donc le seul qui échoue ici.
Les types retenus, mesurés sur la GTX 1050 pour 5,53 s d'audio :

    cuda / int8_float32 : 0,59 s  (RTF 0,107)  <- retenu
    cuda / float32      : 1,08 s  (RTF 0,194)
    cpu  / int8         : 2,07 s  (RTF 0,375)

On essaie les combinaisons dans cet ordre et on s'arrête à la première qui démarre.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

# (device, type de calcul) par ordre de préférence. float16 est volontairement absent.
COMBINAISONS: dict[str, tuple[tuple[str, str], ...]] = {
    "auto": (("cuda", "int8_float32"), ("cuda", "float32"), ("cpu", "int8")),
    "cuda": (("cuda", "int8_float32"), ("cuda", "float32"), ("cpu", "int8")),
    "cpu": (("cpu", "int8"), ("cpu", "float32")),
}

MODELES = ("tiny", "base", "small", "medium", "large-v3")


@dataclass
class Resultat:
    """Ce qu'on retient d'une transcription."""

    texte: str = ""
    langue: str = ""
    probabilite_langue: float = 0.0
    duree_audio_s: float = 0.0
    duree_s: float = 0.0

    @property
    def rtf(self) -> float:
        return self.duree_s / self.duree_audio_s if self.duree_audio_s > 0 else 0.0

    def resume(self) -> str:
        return (
            f"{self.duree_audio_s:.2f} s d'audio transcrites en {self.duree_s:.2f} s "
            f"(RTF {self.rtf:.2f}) — langue {self.langue} ({self.probabilite_langue:.0%})"
        )


class Transcriber:
    """Charge un modèle Whisper une fois, puis transcrit des tableaux numpy."""

    def __init__(
        self,
        modele: str = "small",
        device: str = "auto",
        langue: str = "fr",
        compute: str = "",
        beam_size: int = 5,
    ) -> None:
        self.modele = modele
        self.device_pref = (device or "auto").lower()
        self.langue = langue
        self.compute = compute
        self.beam_size = beam_size
        self.device = ""
        self.compute_reel = ""
        self.erreur: str | None = None
        self._modele = None
        self.duree_s = 0.0

    # ------------------------------------------------------------------ chargement
    @property
    def pret(self) -> bool:
        return self._modele is not None

    def charger(self) -> bool:
        """Charge le modèle à la première combinaison qui fonctionne."""
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            self.erreur = f"faster-whisper non installé : {exc}"
            return False

        essais = COMBINAISONS.get(self.device_pref, COMBINAISONS["auto"])
        if self.device_pref in ("auto", "cuda") and self.compute:
            essais = ((self.device_pref, self.compute),) + essais
        elif self.device_pref == "cpu" and self.compute:
            essais = (("cpu", self.compute),) + essais

        problemes: list[str] = []
        for device, compute in essais:
            try:
                self._modele = WhisperModel(self.modele, device=device, compute_type=compute)
            except Exception as exc:
                problemes.append(f"{device}/{compute} : {str(exc).splitlines()[0][:100]}")
                continue
            self.device, self.compute_reel = device, compute
            return True

        self.erreur = "aucune configuration de whisper n'a démarré — " + " | ".join(problemes)
        return False

    # ------------------------------------------------------------------ synthèse
    def transcrire(self, audio: np.ndarray) -> Resultat:
        """Transcrit un tableau float32 mono 16 kHz."""
        if self._modele is None:
            raise RuntimeError(self.erreur or "modèle non chargé")

        signal = np.asarray(audio, dtype=np.float32).reshape(-1)
        if signal.size == 0:
            return Resultat()

        debut = time.monotonic()
        segments, info = self._modele.transcribe(
            signal,
            language=self.langue or None,  # None = détection automatique
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        texte = " ".join(segment.text.strip() for segment in segments).strip()
        duree = time.monotonic() - debut
        self.duree_s += duree

        return Resultat(
            texte=texte,
            langue=getattr(info, "language", "") or "",
            probabilite_langue=float(getattr(info, "language_probability", 0.0) or 0.0),
            duree_audio_s=signal.size / 16000.0,
            duree_s=duree,
        )

    def transcrire_fichier(self, chemin: str) -> Resultat:
        """Transcrit un fichier audio quelconque (toute fréquence, tout format lisible)."""
        import soundfile as sf

        audio, frequence = sf.read(chemin, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        if frequence != 16000:
            # Rééchantillonnage linéaire : suffisant pour de la parole et sans dépendance.
            n = int(round(mono.size * 16000 / frequence))
            mono = np.interp(
                np.linspace(0, mono.size - 1, n), np.arange(mono.size), mono
            ).astype(np.float32)
        return self.transcrire(mono)
