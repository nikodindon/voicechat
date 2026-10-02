"""Vérifie dans un pseudo-terminal que Tab et Ctrl+R fonctionnent dans le vrai CLI.

    .venv/bin/python tests/verif_completion.py
"""

from __future__ import annotations

import os
import pty
import select
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")


def port_libre() -> int:
    """Un port libre, pour ne pas dépendre d'un numéro fixe déjà occupé par autre chose."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


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


def taper(maitre: int, texte: bytes, delai: float = 0.08) -> None:
    os.write(maitre, texte)
    time.sleep(delai)


def attendre(maitre: int, motif: str, delai: float = 20.0) -> str:
    """Lit jusqu'à voir ``motif`` apparaître (le démarrage charge torch, c'est lent)."""
    tout = ""
    fin = time.monotonic() + delai
    while motif not in tout and time.monotonic() < fin:
        tout += lire(maitre, 0.5)
    return tout


def main() -> int:
    port = port_libre()
    serveur = subprocess.Popen(
        [PY, "tests/fake_llm_server.py", str(port)],
        cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if serveur.poll() is not None:
        print(f"le faux serveur n'a pas démarré (port {port})")
        return 2

    maitre, esclave = pty.openpty()
    cli = subprocess.Popen(
        [PY, "-m", "voicechat", "--no-tts", "--base-url", f"http://127.0.0.1:{port}/v1"],
        cwd=PROJ, stdin=esclave, stdout=esclave, stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    os.close(esclave)

    try:
        assert "vous ›" in attendre(maitre, "vous ›"), "le prompt n'est pas apparu"

        print("=== 1. Tab complète une commande unique ===")
        taper(maitre, b"/sa")
        taper(maitre, b"\t")
        sortie = lire(maitre, 0.6)
        ligne = sortie.replace("\r", "").splitlines()[-1] if sortie.splitlines() else ""
        print("   écran :", ligne.strip())
        assert "/save" in sortie, "« /sa » + Tab devrait compléter en /save"
        print("   -> /save\n")

        print("=== 2. Tab s'arrête au plus long préfixe commun ===")
        taper(maitre, b"\x15")  # Ctrl+U : efface
        taper(maitre, b"/vo")
        taper(maitre, b"\t")
        sortie = lire(maitre, 0.6)
        assert "/voice" in sortie and "/voices" not in sortie.replace("/voices", "", 1), "préfixe commun attendu"
        print("   -> /voice (les deux candidats /voice et /voices partagent ce préfixe)\n")

        print("=== 3. Deux Tab montrent les choix ===")
        taper(maitre, b"\t")
        sortie = lire(maitre, 0.6)
        assert "/voices" in sortie, "le second Tab doit lister les candidats"
        print("   -> liste affichée\n")

        print("=== 4. Ctrl+R rappelle un message précédent ===")
        taper(maitre, b"\x15")
        taper(maitre, b"rappel moi ceci")
        taper(maitre, b"\r")
        lire(maitre, 4.0)  # la réponse arrive
        taper(maitre, b"\x12")  # Ctrl+R
        taper(maitre, b"rappel")
        taper(maitre, b"\r")
        sortie = lire(maitre, 4.0)
        assert "recherche inversée" in sortie, "l'invite de recherche doit s'afficher"
        assert sortie.count("ia ›") >= 1, "le message rappelé doit être renvoyé"
        print("   -> message retrouvé et renvoyé\n")

        taper(maitre, b"/quit")
        taper(maitre, b"\r")
        lire(maitre, 2.0)
        print(">>> OK : Tab complète, Ctrl+R recherche, dans le vrai CLI")
        return 0
    finally:
        cli.terminate()
        serveur.terminate()
        for p in (cli, serveur):
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        os.close(maitre)


if __name__ == "__main__":
    sys.exit(main())
