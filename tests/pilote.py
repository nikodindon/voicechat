"""Pilote de CLI dans un vrai pseudo-terminal, partagé par les scripts de vérification.

Trois choses apprises à la dure, qui expliquent la forme de ce fichier :

1. **Le texte et l'Entrée partent en deux écritures séparées.** Tout envoyer d'un coup
   fait passer la ligne pour un collage, et l'éditeur attend alors une Entrée distincte
   — comportement voulu depuis la v0.1, pour qu'un texte collé ne s'envoie pas tout seul
   (cf. tests/test_editor.py).

2. **On ne peut pas attendre le prompt « vous › » pour savoir qu'un tour est fini.**
   L'éditeur de ligne le réaffiche à chaque frappe : il réapparaît donc dès que la
   question est tapée, bien avant la réponse. On attend soit un silence, soit un motif
   précis quand la commande appelle le modèle sans rien afficher pendant plusieurs
   secondes.
"""

from __future__ import annotations

import os
import pty
import re
import select
import subprocess
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")


def lire(fd: int, duree: float) -> str:
    morceaux: list[bytes] = []
    fin = time.monotonic() + duree
    while time.monotonic() < fin:
        pret, _, _ = select.select([fd], [], [], 0.1)
        if pret:
            try:
                morceaux.append(os.read(fd, 65536))
            except OSError:
                break
    return b"".join(morceaux).decode("utf-8", "replace")


class Session:
    """Un vrai CLI, dans un vrai pseudo-terminal, qui parle à un vrai serveur."""

    def __init__(
        self, environnement: dict[str, str] | None = None, options: list[str] | None = None
    ) -> None:
        self.sortie = ""  # avant tout appel : `attendre()` s'appuie dessus
        self.maitre, esclave = pty.openpty()
        self.processus = subprocess.Popen(
            [PY, "-m", "voicechat", *(options if options is not None else ["--no-tts"])],
            cwd=PROJ,
            stdin=esclave,
            stdout=esclave,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(environnement or {})},
            close_fds=True,
        )
        os.close(esclave)
        self.sortie = self.attendre("vous ›", 90.0)

    def attendre(self, motif: str, delai: float) -> str:
        """Lit jusqu'à voir `motif`, et rend seulement ce qui vient d'arriver.

        Rendre les données **nouvelles** et non le tampon cumulatif est essentiel : le
        tampon contient déjà « vous › » depuis le démarrage, donc chercher dedans ferait
        rendre la main immédiatement, à chaque appel, sans rien lire du tout.
        """
        recu = ""
        fin = time.monotonic() + delai
        while motif not in recu and time.monotonic() < fin:
            recu += lire(self.maitre, 0.5)
        self.sortie += recu
        return recu

    def attendre_calme(self, delai_max: float, calme: float = 2.0, motif: str = "") -> str:
        """Lit jusqu'à ce que la sortie se taise — ou jusqu'à voir `motif`."""
        recu = ""
        dernier = time.monotonic()
        fin = time.monotonic() + delai_max
        while time.monotonic() < fin:
            morceau = lire(self.maitre, 0.25)
            if morceau:
                recu += morceau
                dernier = time.monotonic()
            if motif:
                if motif in recu:
                    break
            elif time.monotonic() - dernier >= calme:
                break
        self.sortie += recu
        return recu

    def envoyer(self, ligne: str, delai: float = 240.0, motif: str = "") -> str:
        """Tape une ligne, l'envoie, et attend la fin du traitement."""
        os.write(self.maitre, ligne.encode("utf-8"))
        time.sleep(0.06)
        os.write(self.maitre, b"\r")
        return self.attendre_calme(delai, motif=motif)

    def fermer(self) -> None:
        self.processus.terminate()
        try:
            self.processus.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.processus.kill()
        os.close(self.maitre)


def propre(texte: str) -> str:
    """Sortie de terminal débarrassée des séquences d'échappement et des retours chariot."""
    return re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", texte).replace("\r", "")
