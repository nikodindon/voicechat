"""Vérifie de bout en bout ce qui part RÉELLEMENT à la synthèse (v0.5).

Pourquoi ce script : lire la réponse à l'écran ne prouve rien sur ce que la voix
prononce. On lance donc le vrai CLI, avec le vrai Kokoro, contre un faux serveur
qui renvoie du markdown, et on capture la trace `[tts]` émise juste avant chaque
synthèse (`VOICECHAT_TRACE_TTS=1`).

Le stderr est capturé par un tuyau séparé du pseudo-terminal : la réponse streamée
et les traces ne se mélangent donc pas, et on ne peut pas confondre les deux.

    .venv/bin/python tests/verif_nettoyage.py
"""

from __future__ import annotations

import os
import pty
import re
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")

MESSAGE = "Montre-moi un exemple markdown avec une liste, un tableau et du code."


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


def attendre(maitre: int, motif: str, delai: float) -> str:
    tout = ""
    fin = time.monotonic() + delai
    while motif not in tout and time.monotonic() < fin:
        tout += lire(maitre, 0.5)
    return tout


def taper(maitre: int, texte: str, delai: float = 0.15) -> None:
    """Écrit comme un clavier : le texte, puis Entrée dans un paquet séparé.

    Écrire la ligne et l'Entrée d'un seul coup serait pris pour un collage — et un
    collage n'est justement pas envoyé automatiquement (cf. tests/verif_collage.py).
    """
    os.write(maitre, texte.encode())
    time.sleep(delai)
    os.write(maitre, b"\r")


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

    environnement = {
        **os.environ,
        "VOICECHAT_TRACE_TTS": "1",  # trace chaque phrase avant synthèse
        "PULSE_SOURCE": os.environ.get(
            "PULSE_SOURCE", "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
        ),
    }

    maitre, esclave = pty.openpty()
    cli = subprocess.Popen(
        [PY, "-m", "voicechat", "--base-url", f"http://127.0.0.1:{port}/v1"],
        cwd=PROJ, stdin=esclave, stdout=esclave, stderr=subprocess.PIPE,
        env=environnement, close_fds=True,
    )
    os.close(esclave)

    traces: list[str] = []
    autres: list[str] = []
    vues: list[str] = []

    def lire_stderr() -> None:
        assert cli.stderr is not None
        for brute in cli.stderr:
            ligne = brute.decode("utf-8", "replace").rstrip()
            if ligne.startswith("[tts] "):
                traces.append(ligne[6:])
            elif ligne:
                autres.append(ligne)

    fil = threading.Thread(target=lire_stderr, daemon=True)
    fil.start()

    try:
        # Le premier lancement charge torch + Kokoro : c'est long.
        if "vous ›" not in attendre(maitre, "vous ›", 180.0):
            print("le prompt n'est pas apparu (Kokoro n'a pas fini de charger ?)")
            return 2
        print("CLI prêt, Kokoro chargé.\n")

        taper(maitre, MESSAGE)
        # On laisse le temps au flux, à la synthèse et à la lecture.
        vues.append(lire(maitre, 45.0))
    finally:
        cli.terminate()
        try:
            cli.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cli.kill()
        fil.join(timeout=5)
        serveur.terminate()
        serveur.wait(timeout=5)

    if not traces:
        print("aucune trace [tts] : le pipeline vocal n'a rien reçu\n")
        print("--- dernières lignes du terminal ---")
        print("".join(vues)[-2000:].replace("\r", ""))
        print("\n--- dernières lignes de stderr (hors trace) ---")
        for ligne in autres[-25:]:
            print("   ", ligne)
        return 2

    print(f"=== {len(traces)} segments réellement envoyés à la synthèse ===")
    for t in traces:
        print("  ", repr(t))

    tout = "\n".join(traces)
    controles = [
        ("aucun bloc de code prononcé", "powermetrics" not in tout and "bash" not in tout),
        ("aucun backtick", "`" not in tout),
        ("aucune barre de tableau", "|" not in tout),
        ("aucun crochet de lien", "[" not in tout and "]" not in tout),
        ("lien réduit à son texte", "la doc Ubuntu" in tout),
        ("abréviation développée", "Monsieur Dupont" in tout),
        ("pourcentage prononçable", "pour cent" in tout),
        ("degré prononçable", "degrés" in tout),
        ("marqueur de liste retiré", not re.search(r"(?:^|\s)2\.", tout)),
    ]

    print("\n=== contrôles ===")
    echecs = 0
    for nom, ok in controles:
        print(f"  {'OK ' if ok else 'KO '} {nom}")
        echecs += 0 if ok else 1

    print()
    if echecs:
        print(f">>> {echecs} contrôle(s) en échec")
        return 1
    print(">>> OK : ce qui est prononcé ne contient plus de balisage markdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
