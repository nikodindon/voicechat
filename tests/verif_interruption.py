"""Vérifie l'interruption à chaud (Ctrl+C et Échap) sur le vrai CLI, dans un pty.

    .venv/bin/python tests/verif_interruption.py
"""

from __future__ import annotations

import os
import pty
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
PORT = "8097"


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


def taper(maitre: int, texte: str, delai: float = 0.08) -> None:
    """Écrit comme un clavier : le texte, puis Entrée dans un paquet séparé.

    Un vrai terminal envoie une frappe = un paquet ; écrire la ligne et l'Entrée
    d'un seul coup ressemblerait à un collage (et c'est bien ce que le repli détecte).
    """
    os.write(maitre, texte.encode())
    time.sleep(delai)
    os.write(maitre, b"\r")


def main() -> int:
    serveur = subprocess.Popen(
        [PY, "tests/fake_llm_server.py", PORT],
        cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)

    maitre, esclave = pty.openpty()
    cli = subprocess.Popen(
        [PY, "-m", "voicechat", "--no-tts", "--debug",
         "--base-url", f"http://127.0.0.1:{PORT}/v1"],
        cwd=PROJ, stdin=esclave, stdout=esclave, stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    os.close(esclave)

    try:
        lire(maitre, 4.0)  # démarrage
        print("=== 1. Ctrl+C en pleine génération ===")
        taper(maitre, "raconte une longue histoire")
        time.sleep(0.5)  # la réponse est en cours
        cli.send_signal(signal.SIGINT)  # ce que fait la touche Ctrl+C
        sortie = lire(maitre, 3.0)
        print(sortie.replace("\r", "").strip())
        assert "[Ctrl+C] génération, synthèse et lecture coupées" in sortie
        assert "vous ›" in sortie, "la session doit continuer après un Ctrl+C"
        print("   -> coupé, et la session survit\n")

        print("=== 2. Une réponse qui finit normalement ===")
        taper(maitre, "une question courte")
        sortie = lire(maitre, 8.0)
        fin = sortie.replace("\r", "")
        print("\n".join(l for l in fin.splitlines() if l.strip())[-400:])
        assert "Fin du message." in fin, "la réponse complète doit arriver"
        assert "tok" in fin, "les statistiques doivent afficher des tokens"
        print("   -> réponse complète + statistiques en tokens\n")

        print("=== 3. Échap en pleine génération (voix coupée, réponse gardée) ===")
        taper(maitre, "encore une longue histoire")
        time.sleep(0.4)
        os.write(maitre, b"\x1b")  # Échap
        sortie = lire(maitre, 6.0)
        fin = sortie.replace("\r", "")
        print("\n".join(l for l in fin.splitlines() if l.strip())[-400:])
        assert "[Échap] voix coupée" in fin
        assert "Fin du message." in fin, "Échap ne doit PAS couper la génération"
        print("   -> voix coupée, génération poursuivie jusqu'au bout\n")

        taper(maitre, "/quit")
        lire(maitre, 2.0)
        print(">>> OK : Ctrl+C coupe tout, Échap ne coupe que la voix")
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
