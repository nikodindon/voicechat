"""Vérifie la précédence de configuration et les réglages par modèle (v1.0).

Ordre promis : **arguments > environnement > fichier TOML > valeurs codées en dur**.
On le vérifie sur le vrai chemin de code (`config_from_args`, celui que le CLI
utilise), puis en lançant le vrai CLI contre le vrai serveur pour voir la ligne des
réglages par modèle apparaître.

    .venv/bin/python tests/verif_config.py
"""

from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import tempfile
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


def attendre(fd: int, motif: str, delai: float) -> str:
    tout = ""
    fin = time.monotonic() + delai
    while motif not in tout and time.monotonic() < fin:
        tout += lire(fd, 0.5)
    return tout


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-config-"))
    chemin = dossier / "config.toml"

    # Le vrai nom renvoyé par llama.cpp contient des points : en TOML il faut le
    # mettre entre guillemets, sinon chaque point ouvre une sous-table.
    modele = "Ornith-1.5-35B-A3B-APEX-i-mini"
    chemin.write_text(
        f"""\
[voicechat]
voice = "af_heart"
speed = 1.25
temperature = 0.9

[modeles."{modele}"]
voice = "ff_siwis:3+ef_dora:1"
temperature = 0.2
""",
        encoding="utf-8",
    )

    print("=" * 78)
    print("Configuration : précédence et réglages par modèle")
    print("=" * 78)
    print(f"fichier TOML : {chemin}")
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        print(f"  | {ligne}")
    print()

    from voicechat import cli
    from voicechat.config import nom_court

    # ---------------------------------------------------------------- partie A
    print("--- A. précédence, sur le vrai chemin de code ---")
    os.environ["VOICECHAT_CONFIG"] = str(chemin)
    os.environ.pop("VOICECHAT_VOICE", None)
    os.environ.pop("VOICECHAT_SPEED", None)

    args = cli.build_parser().parse_args([])  # aucun argument fourni
    cfg = cli.config_from_args(args)
    print(f"  TOML seul          : voice={cfg.voice} speed={cfg.speed} "
          f"temperature={cfg.temperature}")
    ok_toml = cfg.voice == "af_heart" and cfg.speed == 1.25 and cfg.temperature == 0.9

    os.environ["VOICECHAT_VOICE"] = "if_sara"
    cfg = cli.config_from_args(args)
    print(f"  + env VOICE=if_sara: voice={cfg.voice} (le TOML disait af_heart)")
    ok_env = cfg.voice == "if_sara"

    args2 = cli.build_parser().parse_args(["--voice", "bm_george", "--speed", "0.9"])
    cfg = cli.config_from_args(args2)
    print(f"  + args --voice     : voice={cfg.voice} speed={cfg.speed} "
          f"(l'env disait if_sara, le TOML 1.25)")
    ok_args = cfg.voice == "bm_george" and cfg.speed == 0.9

    os.environ.pop("VOICECHAT_VOICE", None)

    # réglages par modèle
    cfg = cli.config_from_args(args)
    ajustee, ajustes = cfg.pour_modele(f"/mnt/data/sdc2/models/{modele}.gguf")
    print(f"  pour_modele        : {ajustes}")
    ok_modele = ajustes.get("voice") == "ff_siwis:3+ef_dora:1" and ajustes.get(
        "temperature"
    ) == 0.2
    print(f"  nom court attendu  : {nom_court('.../' + modele + '.gguf')}")
    print()

    # ---------------------------------------------------------------- partie B
    print("--- B. le vrai CLI affiche-t-il les réglages par modèle ? ---")
    maitre, esclave = pty.openpty()
    processus = subprocess.Popen(
        [PY, "-m", "voicechat", "--no-tts"],
        cwd=PROJ, stdin=esclave, stdout=esclave, stderr=subprocess.STDOUT,
        env={**os.environ, "VOICECHAT_CONFIG": str(chemin)},
        close_fds=True,
    )
    os.close(esclave)
    try:
        sortie = attendre(maitre, "vous ›", 60.0)
    finally:
        processus.terminate()
        try:
            processus.wait(timeout=10)
        except subprocess.TimeoutExpired:
            processus.kill()

    propre = sortie.replace("\r", "")
    for ligne in propre.splitlines():
        if "réglages" in ligne or "voicechat " in ligne:
            print(f"  {ligne.strip()}")
    ok_visible = "réglages du fichier de configuration" in propre
    print()

    controles = [
        ("le TOML fournit les valeurs par défaut", ok_toml),
        ("l'environnement surclasse le TOML", ok_env),
        ("les arguments surclassent l'environnement", ok_args),
        ("les réglages par modèle sont trouvés", ok_modele),
        ("le CLI annonce les réglages appliqués", ok_visible),
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
    print(">>> OK : la précédence est celle promise, et les réglages par modèle arrivent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
