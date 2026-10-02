"""Compare les façons de capturer le micro à une référence externe (`parec`).

Pourquoi ce banc d'essai : sur cette machine, une phrase jouée dans les haut-parleurs
est transcrite correctement quand on la capture avec `parec`, et mal quand on la
capture avec PortAudio/sounddevice. La parole est alors hachée (« la lumière du
salon » → « l'alignement du salaud »), ce qui ressemble à des échantillons perdus.

Comme la source est une boucle numérique (sortie « monitor » de PulseAudio), deux
clients honnêtes doivent capter **exactement** le même signal : la corrélation entre
eux doit être proche de 1. Toute méthode qui s'en écarte est fautive, et on le mesure
au lieu de le supposer.

    python tests/bench_capture.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PHRASE = "Allume la lumière du salon et vérifie que la porte est fermée."
SORTIE_MONITOR = "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
DUREE = 6.0


def source_monitor() -> str:
    """La sortie « monitor » correspondant au puits par défaut."""
    try:
        puits = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
        if puits:
            return f"{puits}.monitor"
    except Exception:
        pass
    return SORTIE_MONITOR


def demarrer_reference() -> tuple[subprocess.Popen, object]:
    fichier = open("/tmp/vc_ref.raw", "wb")
    proc = subprocess.Popen(
        ["parec", "--device=" + source_monitor(), "--format=s16le",
         "--rate=16000", "--channels=1"],
        stdout=fichier, stderr=subprocess.DEVNULL,
    )
    return proc, fichier


def arreter_reference(proc, fichier) -> np.ndarray:
    time.sleep(0.3)
    proc.terminate()
    proc.wait(timeout=5)
    fichier.close()
    return np.fromfile("/tmp/vc_ref.raw", dtype="<i2").astype(np.float32) / 32768.0


def correlation(mien: np.ndarray, ref: np.ndarray) -> float:
    """Corrélation au meilleur alignement (décalage par corrélation croisée)."""
    n = min(mien.size, ref.size)
    if n < 1000:
        return 0.0
    a = mien[:n] - mien[:n].mean()
    b = ref[:n] - ref[:n].mean()
    c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)))
    d = int(np.argmax(c))
    if d > n:
        d -= 2 * n
    if d >= 0:
        x, y = mien, ref[d : d + mien.size]
    else:
        x, y = mien[-d:], ref
    k = min(x.size, y.size)
    return float(np.corrcoef(x[:k], y[:k])[0, 1]) if k > 1000 else 0.0


def capturer_avec(nom: str, budget: float = DUREE) -> np.ndarray | None:
    """Capture avec la méthode demandée, pendant que parec capture en parallèle."""
    import sounddevice as sd

    from voicechat.micro import Microphone

    res: dict = {}

    def travail() -> None:
        if nom == "microphone":
            micro = Microphone()
            blocs = []
            with micro:
                debut = time.monotonic()
                while time.monotonic() - debut < budget:
                    bloc = micro.lire(0.5)
                    if bloc is None:
                        break
                    blocs.append(bloc)
            res["x"] = np.concatenate(blocs) if blocs else None
        elif nom == "sd.rec":
            res["x"] = np.asarray(
                sd.rec(int(budget * 16000), samplerate=16000, channels=1,
                       dtype="float32", device="pulse")
            ).reshape(-1)
            sd.wait()
        else:
            raise ValueError(nom)

    fil = threading.Thread(target=travail, daemon=True)
    fil.start()
    time.sleep(1.0)
    sd.play(SIG, samplerate=24000, device="pulse")
    sd.wait()
    fil.join(timeout=budget + 10)
    return res.get("x")


def main() -> int:
    import soundfile as sf

    print("=" * 72)
    print("Comparaison des méthodes de capture (référence : parec)")
    print("=" * 72)
    try:
        from voicechat.tts import KokoroTTS

        tts = KokoroTTS(voice="ff_siwis", lang_code="f", device="auto")
        tts.load()
        audio = tts.synth(PHRASE)
        sf.write("/tmp/vc_phrase.wav", audio, 24000)
        print(f"phrase : « {PHRASE} » (synthétisée, {len(audio)/24000:.2f} s)")
    except Exception:
        audio, _ = sf.read("/tmp/vc_phrase.wav", dtype="float32")
        print(f"phrase : « {PHRASE} » (relue depuis /tmp/vc_phrase.wav)")

    global SIG
    SIG = (audio * 0.5).astype(np.float32)

    resultats: list[tuple[str, float]] = []
    for nom in ("microphone", "sd.rec"):
        proc, fichier = demarrer_reference()
        time.sleep(0.3)
        try:
            capture = capturer_avec(nom)
        finally:
            reference = arreter_reference(proc, fichier)
        if capture is None or capture.size < 1000:
            print(f"  {nom:14} : capture vide")
            continue
        c = correlation(capture, reference)
        resultats.append((nom, c))
        verdict = "OK" if c > 0.9 else "DÉGRADÉ"
        print(f"  {nom:14} : {capture.size / 16000:5.2f} s | corrélation {c:+.4f}  {verdict}")

    print()
    print("Sur cette machine : parec est propre, PortAudio hache la parole.")
    print("La corrélation doit valoir ~1 : la source est une boucle numérique,")
    print("donc les deux clients reçoivent le même signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
