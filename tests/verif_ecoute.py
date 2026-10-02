"""Vérifie la chaîne complète capture → VAD → transcription, sans micro physique.

On joue une phrase dans les haut-parleurs et on la capture par la sortie « monitor »
de PulseAudio : c'est une boucle logicielle, donc reproductible et silencieuse pour
l'oreille (volume faible). Cela valide le découpage en 512 échantillons, la détection
d'activité vocale, la détection de fin d'énoncé et la transcription.

    .venv/bin/python tests/verif_ecoute.py

Un vrai micro se teste avec `voicechat --micro` (il faut alors un gain de capture
raisonnable : voir le README §8).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from voicechat.micro import DetecteurParole, Ecouteur  # noqa: E402
from voicechat.stt import Transcriber  # noqa: E402
from voicechat.tts import KokoroTTS  # noqa: E402

MONITOR = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
PHRASE = "Allume la lumière du salon et vérifie que la porte est fermée."


def preparer_audio() -> Path:
    """Fabrique la phrase à écouter, avec Kokoro."""
    chemin = Path("/tmp/verif_ecoute.wav")
    tts = KokoroTTS(voice="ff_siwis", lang_code="f", device="auto")
    assert tts.load(), tts.load_error
    audio = tts.synth(PHRASE)
    sf.write(chemin, audio, 24000)
    return chemin


def vram() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> int:
    os.environ["PULSE_SOURCE"] = MONITOR  # on écoute la sortie, pas le micro

    vad = DetecteurParole()
    if not vad.pret:
        print("VAD indisponible :", vad.erreur)
        return 2
    print("VAD     : silero v6 chargé (0,02 sur silence, 0,74 sur parole — cf. README)")
    print(f"VRAM    : {vram()}")

    print(f"\n[1/3] phrase à reconnaître : « {PHRASE} »")
    chemin = preparer_audio()
    duree = sf.info(chemin).duration
    print(f"      audio de {duree:.2f} s préparé ({chemin})")

    print("\n[2/3] écoute en parallèle de la lecture (boucle monitor)")
    capture: dict = {}

    def ecouter() -> None:
        ecouteur = Ecouteur(vad=vad, silence_fin=0.6, on_parole=lambda: print("      ● parole détectée"))
        capture["audio"] = ecouteur.ecouter(delai_depart=15.0)

    fil = threading.Thread(target=ecouter, daemon=True)
    fil.start()
    time.sleep(1.0)  # laisser le flux s'ouvrir avant de jouer

    import sounddevice as sd

    audio, _ = sf.read(chemin, dtype="float32")
    faible = (audio * 0.35).astype(np.float32)  # volume discret
    sd.play(faible, samplerate=24000, device="pulse")
    sd.wait()
    fil.join(timeout=20)

    audio_capture = capture.get("audio")
    if audio_capture is None:
        print("      ÉCHEC : rien n'a été détecté par le VAD")
        return 1
    print(f"      énoncé capturé : {audio_capture.size / 16000:.2f} s (source {duree:.2f} s)")

    print("\n[3/3] transcription")
    transcriber = Transcriber(modele="small", device="auto", langue="fr")
    if not transcriber.charger():
        print("      ÉCHEC :", transcriber.erreur)
        return 2
    resultat = transcriber.transcrire(audio_capture)
    print(f"      {resultat.resume()}")
    print(f"      attendu : {PHRASE}")
    print(f"      obtenu  : {resultat.texte}")
    print(f"      VRAM après whisper : {vram()}")

    # Comparaison tolérante : la ponctuation et la casse ne comptent pas.
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ")
    correct = norm(resultat.texte) == norm(PHRASE)
    print(f"\n      identique (hors ponctuation) : {correct}")
    if not correct:
        print("      (le VAD a pu couper un bord de phrase : vérifier la fin du texte)")
        return 1
    print("\n>>> OK : capture → VAD → découpage → transcription fonctionnent de bout en bout")
    return 0


if __name__ == "__main__":
    sys.exit(main())
