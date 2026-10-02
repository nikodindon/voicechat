"""Vérifie que la suite de tests ne dépend PAS des paquets lourds (v1.1).

Pourquoi c'est un test à part entière : si un jour quelqu'un écrit ``import torch`` en
haut d'un module au lieu de l'importer dans la fonction qui en a besoin, la suite
deviendrait impossible à lancer sans 3 Go de dépendances et un GPU. Ce script attrape
cette régression tout de suite, au lieu de la laisser pourrir la CI six mois plus tard.

Utilisation (c'est exactement ce que fait la CI) :

    python tests/verif_sans_lourds.py

Attention au détail qui compte : un paquet réellement absent lève
``ModuleNotFoundError``, pas ``ImportError``. Depuis pytest 8.2,
``pytest.importorskip`` ne saute **que** sur ``ModuleNotFoundError``. Lever un
``ImportError`` nu ferait donc échouer les tests au lieu de les ignorer, et ce script
mentirait sur ce qui se passe dans une vraie CI sans ces paquets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent

# Paquets dont la suite doit pouvoir se passer : ils ne servent qu'à la voix, à la
# transcription et au GPU, jamais à la logique testée hors ligne.
LOURDS = ("torch", "kokoro", "faster_whisper", "ctranslate2", "av")


class Bloqueur:
    """Empêche l'import des paquets lourds, comme s'ils n'étaient pas installés."""

    def find_spec(self, nom, chemin=None, cible=None):
        if nom.split(".")[0] in LOURDS:
            raise ModuleNotFoundError(f"No module named '{nom}'", name=nom)
        return None


def verifier_blocage() -> bool:
    """Sans ce contrôle, des tests « verts » ne prouveraient rien du tout."""
    for paquet in LOURDS:
        try:
            __import__(paquet)
        except ModuleNotFoundError:
            continue
        print(f"!! « {paquet} » reste importable : le blocage ne fonctionne pas")
        return False
    return True


def lancer_pytest() -> int:
    """Le vrai travail : la suite complète, avec les paquets lourds interdits."""
    sys.meta_path.insert(0, Bloqueur())
    if not verifier_blocage():
        return 3

    print(f"blocage actif : {', '.join(LOURDS)}")
    print("la CI n'installe que numpy, soundfile et pytest : si la suite passe ici, elle y passe.\n")

    import pytest

    return pytest.main([str(PROJ / "tests"), "-q"])


def main() -> int:
    # En sous-processus : les modules que ce script importe pour lui-même le resteraient
    # sinon en mémoire, et le blocage n'aurait plus aucun effet sur eux.
    resultat = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--sous-processus"],
        cwd=PROJ,
        check=False,
    )
    return resultat.returncode


if __name__ == "__main__":
    if "--sous-processus" in sys.argv:
        raise SystemExit(lancer_pytest())
    raise SystemExit(main())
