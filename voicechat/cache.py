"""Réserve disque des phrases déjà synthétisées.

Pourquoi : un LLM se répète énormément — « Bien sûr ! », « Voici les points qui
comptent : », « N'hésite pas si tu as d'autres questions. ». Or la synthèse est
l'étape la plus lente d'un tour. Garder le son de ces phrases les rend instantanées.

Le piège d'un cache audio, c'est la clé : l'audio dépend de la voix, de la vitesse
**et** de la langue, pas seulement du texte. Changer de voix ne doit donc jamais
resservir l'audio de l'ancienne. C'est pour ça que `cle()` prend les quatre.

Le format est du WAV float32 : relisible par n'importe quel lecteur, donc
inspectable à la main — et à l'oreille — si un doute apparaît.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
MAX_FICHIERS = 2000


def dossier_cache() -> Path:
    """``$VOICECHAT_CACHE_DIR``, sinon ``$XDG_CACHE_HOME/voicechat/tts``.

    L'activation, elle, se règle avec ``VOICECHAT_CACHE`` (0/1) — deux variables
    distinctes pour deux choses distinctes, plutôt qu'un nom à une lettre près.
    """
    force = os.environ.get("VOICECHAT_CACHE_DIR")
    if force:
        return Path(force).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    racine = Path(base).expanduser() if base else Path.home() / ".cache"
    return racine / "voicechat" / "tts"


class CacheAudio:
    """Un fichier WAV par phrase, nommé par l'empreinte de la clé.

    Volontairement bête : pas d'index, pas de base de données. Le système de
    fichiers *est* l'index, et une entrée corrompue se rattrape toute seule
    (elle est supprimée et on resynthétise).
    """

    def __init__(
        self,
        dossier: str | Path | None = None,
        max_fichiers: int = MAX_FICHIERS,
        actif: bool = True,
    ) -> None:
        self.dossier = Path(dossier) if dossier else dossier_cache()
        self.max_fichiers = max(1, int(max_fichiers))
        self.actif = actif
        self.hits = 0
        self.misses = 0
        self.ecrits = 0
        self.servis_s = 0.0  # durée d'audio servie depuis la réserve

    # ------------------------------------------------------------------ la clé
    @staticmethod
    def cle(texte: str, voix: str, vitesse: float, langue: str) -> str:
        """Empreinte de ce qui détermine l'audio produit.

        Le texte est pris **après** nettoyage (c'est ce que la synthèse reçoit), et
        la voix entre telle quelle (donc un mélange pondéré a sa propre clé).
        """
        brut = f"{langue}|{voix}|{vitesse:.4f}|{texte}".encode("utf-8")
        return hashlib.sha1(brut).hexdigest()

    def chemin(self, cle: str) -> Path:
        return self.dossier / f"{cle}.wav"

    # --------------------------------------------------------------- lecture
    def lire(self, cle: str) -> np.ndarray | None:
        """L'audio en réserve, ou ``None``. Ne lève jamais."""
        if not self.actif:
            return None
        chemin = self.chemin(cle)
        if not chemin.is_file():
            self.misses += 1
            return None
        try:
            import soundfile as sf

            audio, _ = sf.read(str(chemin), dtype="float32")
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception:
            # Entrée illisible (fichier tronqué, format inattendu) : on la jette et
            # on laisse la synthèse refaire le travail. Un cache ne doit pas casser
            # l'application.
            self._supprimer(chemin)
            self.misses += 1
            return None
        if audio.size == 0:
            self._supprimer(chemin)
            self.misses += 1
            return None
        self.hits += 1
        self.servis_s += audio.size / SAMPLE_RATE
        # On rafraîchit la date : l'élagage retire les plus anciens, donc une phrase
        # souvent réutilisée ne doit pas être évincée par des phrases récentes.
        try:
            os.utime(chemin, None)
        except OSError:
            pass
        return audio

    # --------------------------------------------------------------- écriture
    def ecrire(self, cle: str, audio: np.ndarray) -> bool:
        """Range un audio en réserve. Ne lève jamais (un cache en échec est bénin)."""
        if not self.actif or audio is None or audio.size == 0:
            return False
        try:
            import soundfile as sf

            self.dossier.mkdir(parents=True, exist_ok=True)
            # Écriture atomique : un processus tué en pleine écriture ne doit pas
            # laisser un WAV tronqué que la prochaine lecture prendrait pour bon.
            #
            # `format="WAV"` est obligatoire : soundfile déduit le format de
            # l'extension, et il ne connaît pas « .tmp ». Sans ce paramètre,
            # l'écriture échouait — en silence, puisque `ecrire` ne lève jamais.
            temporaire = self.chemin(cle).with_suffix(".tmp")
            sf.write(
                str(temporaire),
                np.asarray(audio, dtype=np.float32),
                SAMPLE_RATE,
                subtype="FLOAT",
                format="WAV",
            )
            os.replace(temporaire, self.chemin(cle))
        except Exception:
            return False
        self.ecrits += 1
        self._elaguer()
        return True

    # ---------------------------------------------------------------- élagage
    def _supprimer(self, chemin: Path) -> None:
        try:
            chemin.unlink()
        except OSError:
            pass

    def _elaguer(self) -> None:
        """Ramène le dossier sous ``max_fichiers``, en retirant les plus anciens."""
        try:
            entrees = [p for p in self.dossier.iterdir() if p.suffix == ".wav"]
        except OSError:
            return
        if len(entrees) <= self.max_fichiers:
            return
        try:
            entrees.sort(key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for chemin in entrees[: len(entrees) - self.max_fichiers]:
            self._supprimer(chemin)

    # ------------------------------------------------------------------- stats
    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def ratio(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def resume(self) -> str:
        if not self.actif:
            return "réserve désactivée"
        if not self.total:
            return "réserve vide (aucune phrase encore)"
        return (
            f"{self.hits} reprise(s) / {self.total} phrase(s) "
            f"({self.ratio:.0%}), {self.servis_s:.1f} s d'audio resservies"
        )

    def vider(self) -> int:
        """Supprime toutes les entrées. Renvoie le nombre de fichiers retirés."""
        try:
            entrees = list(self.dossier.glob("*.wav"))
        except OSError:
            return 0
        for chemin in entrees:
            self._supprimer(chemin)
        return len(entrees)
