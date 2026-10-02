"""Vérifie la bascule réseau dans le vrai CLI (v0.6).

Le scénario : la cible principale est éteinte, un secours répond. L'utilisateur doit
obtenir sa réponse **et** savoir qu'elle ne vient pas de la machine attendue — une
bascule silencieuse serait un piège (on croit parler au gros serveur et c'est un
autre qui répond).

    .venv/bin/python tests/verif_reprise.py
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


def attendre(fd: int, motif: str, delai: float) -> str:
    tout = ""
    fin = time.monotonic() + delai
    while motif not in tout and time.monotonic() < fin:
        tout += lire(fd, 0.5)
    return tout


def taper(maitre: int, texte: str, delai: float = 0.15) -> None:
    os.write(maitre, texte.encode())
    time.sleep(delai)
    os.write(maitre, b"\r")


def main() -> int:
    port_mort = port_libre()  # réservé puis libéré : personne n'écoute
    port_secours = port_libre()
    serveur = subprocess.Popen(
        [PY, "tests/fake_llm_server.py", str(port_secours)],
        cwd=PROJ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if serveur.poll() is not None:
        print("le faux serveur n'a pas démarré")
        return 2

    principal = f"http://127.0.0.1:{port_mort}/v1"
    secours = f"http://127.0.0.1:{port_secours}/v1"
    print("=" * 78)
    print("Bascule réseau dans le vrai CLI")
    print("=" * 78)
    print(f"cible principale : {principal}   (rien n'écoute)")
    print(f"secours          : {secours}   (faux serveur)")
    print()

    maitre, esclave = pty.openpty()
    cli = subprocess.Popen(
        [PY, "-m", "voicechat", "--no-tts",
         "--base-url", principal, "--secours", secours],
        cwd=PROJ, stdin=esclave, stdout=esclave, stderr=subprocess.STDOUT,
        close_fds=True,
    )
    os.close(esclave)

    try:
        debut = attendre(maitre, "vous ›", 60.0)
        if "vous ›" not in debut:
            print("le prompt n'est pas apparu")
            return 2
        print("--- démarrage ---")
        print(debut.replace("\r", "").strip())
        print()

        taper(maitre, "raconte-moi une histoire courte")
        sortie = lire(maitre, 12.0)
    finally:
        cli.terminate()
        try:
            cli.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cli.kill()
        serveur.terminate()
        serveur.wait(timeout=5)

    propre = sortie.replace("\r", "")
    print("--- après la question ---")
    print(propre.strip())
    print()

    controles = [
        ("la bascule est annoncée", "[réseau]" in propre),
        ("le secours est nommé", secours in propre),
        ("l'URL fautive est nommée", principal in propre),
        ("une réponse est arrivée", "réponse de test" in propre),
        ("aucune erreur bloquante", "[ERREUR]" not in propre),
    ]
    print("=== contrôles ===")
    echecs = 0
    for nom, ok in controles:
        print(f"  {'OK ' if ok else 'KO '} {nom}")
        echecs += 0 if ok else 1
    print()
    if echecs:
        print(f">>> {echecs} contrôle(s) en échec")
        return 1
    print(">>> OK : le secours a servi, et l'utilisateur l'a su")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
