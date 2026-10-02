"""Mesure le gain réel de la réserve d'audio (v1.0).

On synthétise deux fois la même liste de phrases — une fois réserve vide, une fois
réserve pleine — et on compare. Les phrases sont choisies comme celles qu'un LLM
ressert en boucle.

    .venv/bin/python tests/bench_cache.py
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Des tournures qu'un assistant local répète vraiment d'une réponse à l'autre.
PHRASES = [
    "Bien sûr !",
    "Voici les points qui comptent :",
    "N'hésite pas si tu as d'autres questions.",
    "En résumé, il y a deux choses à retenir.",
    "Le mode économie limite la puissance du processeur.",
    "En résumé, il y a deux choses à retenir.",  # répétition volontaire
    "Bien sûr !",  # répétition volontaire
]


def main() -> int:
    from voicechat.cache import CacheAudio
    from voicechat.tts import KokoroTTS

    dossier = Path(tempfile.mkdtemp(prefix="vc-bench-cache-"))
    reserve = CacheAudio(dossier)
    tts = KokoroTTS(voice="ff_siwis", device="auto", cache=reserve)

    print("=" * 78)
    print("Gain de la réserve d'audio (Kokoro réel)")
    print("=" * 78)
    if not tts.load():
        print(f"TTS indisponible : {tts.load_error}")
        return 2
    print(f"Kokoro sur {tts.device} ({tts.device_reason})")
    print(f"réserve : {dossier}")
    print(f"{len(PHRASES)} phrases, dont 2 répétitions volontaires")
    print()

    # --- passage 1 : réserve vide, tout est synthétisé ---------------------
    print("[1/2] réserve vide — tout est synthétisé")
    froids: list[float] = []
    for phrase in PHRASES:
        t0 = time.monotonic()
        audio = tts.synth(phrase)
        froids.append(time.monotonic() - t0)
        print(f"    {len(audio) / 24000:5.2f} s d'audio en {froids[-1] * 1000:6.0f} ms "
              f"— {phrase[:44]}")

    # --- passage 2 : réserve pleine ---------------------------------------
    print()
    print("[2/2] réserve pleine — relecture depuis le disque")
    chauds: list[float] = []
    identiques = True
    for phrase in PHRASES:
        t0 = time.monotonic()
        audio = tts.synth(phrase)
        chauds.append(time.monotonic() - t0)
        print(f"    {len(audio) / 24000:5.2f} s d'audio en {chauds[-1] * 1000:6.0f} ms "
              f"— {phrase[:44]}")

    # --- vérification : l'audio resservi est identique à une synthèse fraîche ----
    servie = tts.synth(PHRASES[0])  # vient de la réserve
    reserve.actif = False  # réserve en pause
    fraiche = tts.synth(PHRASES[0])  # synthèse réelle
    reserve.actif = True
    if servie.shape == fraiche.shape:
        diff = float(abs(servie - fraiche).max())
    else:
        diff = float("inf")
        identiques = False

    total_froid = sum(froids)
    total_chaud = sum(chauds)
    print()
    print("=" * 78)
    print(f"total à froid : {total_froid * 1000:7.0f} ms   "
          f"(médiane {statistics.median(froids) * 1000:.0f} ms/phrase)")
    print(f"total à chaud : {total_chaud * 1000:7.0f} ms   "
          f"(médiane {statistics.median(chauds) * 1000:.0f} ms/phrase)")
    if total_chaud > 0:
        print(f"facteur       : {total_froid / total_chaud:.0f}× plus rapide")
    print(f"réserve       : {reserve.resume()}")
    print(f"~{tts.temps_economise_s:.2f} s de synthèse évitées (estimation)")
    print()
    print(f"Écart max entre le premier passage et le rejeu : {diff:.2e}")
    print("(0 veut dire que l'audio resservi est identique au bit près)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
