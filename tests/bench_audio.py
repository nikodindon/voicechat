"""Mesure le surcoût de lecture par tampon : flux persistant vs sd.play().

    .venv/bin/python tests/bench_audio.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicechat.audio import Speaker  # noqa: E402

N = 10  # nombre de tampons
DUREE = 1.0  # secondes d'audio par tampon
BLOC = np.full(int(24000 * DUREE), 0.001, dtype=np.float32)  # quasi inaudible


def main() -> int:
    try:
        import sounddevice as sd
    except Exception as exc:
        print("sounddevice indisponible :", exc)
        return 1

    print(f"{N} tampons de {DUREE:.0f} s = {N * DUREE:.0f} s d'audio à jouer\n")

    # --- 1. flux persistant (le nouveau chemin) ---
    hp = Speaker()
    hp.start()
    if not hp.pret:
        print("flux persistant INDISPONIBLE :", hp.last_error)
        return 1
    debut = time.monotonic()
    for _ in range(N):
        hp.play(BLOC)
    hp.wait()
    duree_persistant = time.monotonic() - debut
    joue = hp.played_s
    hp.close()
    print(f"flux persistant   : {duree_persistant:.3f} s de temps réel  "
          f"(surcoût {duree_persistant - N * DUREE:+.3f} s)  |  joué {joue:.2f} s")

    # --- 2. ancienne méthode : un flux ouvert/fermé par tampon ---
    debut = time.monotonic()
    for _ in range(N):
        sd.play(BLOC, samplerate=24000)
        sd.wait()
    duree_par_tampon = time.monotonic() - debut
    print(f"sd.play par tampon: {duree_par_tampon:.3f} s de temps réel  "
          f"(surcoût {duree_par_tampon - N * DUREE:+.3f} s)")

    ecart = (duree_par_tampon - duree_persistant) / N * 1000
    print(f"\n>>> environ {ecart:.1f} ms de silence économisées par phrase")
    return 0


if __name__ == "__main__":
    sys.exit(main())
