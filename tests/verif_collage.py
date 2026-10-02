"""Vérifie de bout en bout le collage multi-lignes, en pilotant le vrai CLI dans un pty.

Ce n'est pas un test pytest (il démarre un serveur et un processus) : c'est un script
de vérification manuelle, à lancer depuis la racine du projet.

    .venv/bin/python tests/verif_collage.py

Attendu : 0 réponse après le collage, 1 seule après la frappe d'Entrée.
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
    """Lit tout ce qui arrive sur ``fd`` pendant ``duree`` secondes."""
    morceaux: list[bytes] = []
    fin = time.monotonic() + duree
    while time.monotonic() < fin:
        pret, _, _ = select.select([fd], [], [], 0.2)
        if pret:
            try:
                morceaux.append(os.read(fd, 65536))
            except OSError:
                break
    return b"".join(morceaux).decode("utf-8", "replace")


def main() -> int:
    port = port_libre()
    serveur = subprocess.Popen(
        [PY, "tests/fake_llm_server.py", str(port)],
        cwd=PROJ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if serveur.poll() is not None:
        print(f"le faux serveur n'a pas démarré (port {port})")
        return 2

    maitre, esclave = pty.openpty()
    cli = subprocess.Popen(
        [PY, "-m", "voicechat", "--no-tts", "--base-url", f"http://127.0.0.1:{port}/v1"],
        cwd=PROJ,
        stdin=esclave,
        stdout=esclave,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    os.close(esclave)

    try:
        demarrage = lire(maitre, 4.0)
        print("=== démarrage ===")
        for ligne in demarrage.splitlines():
            if ligne.strip():
                print("   ", ligne.strip())
        assert "vous ›" in demarrage, "le prompt n'est pas apparu"

        # --- un collage de 3 lignes, balisé comme le fait un vrai terminal ---
        COLLAGE = "Question sur trois lignes.\nDeuxieme ligne du collage.\nTroisieme ligne."
        os.write(maitre, f"\x1b[200~{COLLAGE}\x1b[201~".encode())
        apres_collage = lire(maitre, 2.0)

        print("\n=== après le collage, AVANT Entrée ===")
        messages = apres_collage.count("ia ›")
        print(f"    réponses du modèle : {messages}   (attendu : 0)")
        assert "3 lignes" in apres_collage, "le résumé du collage devrait s'afficher"
        assert messages == 0, "un collage ne doit rien envoyer"

        # --- on appuie sur Entrée : le message part, en un seul morceau ---
        os.write(maitre, b"\r")
        apres_entree = lire(maitre, 6.0)

        print("\n=== après Entrée ===")
        messages = apres_entree.count("ia ›")
        print(f"    réponses du modèle : {messages}   (attendu : 1)")
        extrait = apres_entree.strip().replace("\r", "")[:120].replace("\n", " | ")
        print("    extrait :", extrait)
        assert messages == 1, "un collage de 3 lignes doit produire UNE seule réponse"

        os.write(maitre, b"/quit\r")
        lire(maitre, 2.0)

        print("\n>>> OK : le texte collé est parti en UN SEUL message")
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
