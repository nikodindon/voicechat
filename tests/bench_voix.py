"""Mesure ce que valent les mélanges de voix Kokoro — en français.

Ce que ce banc d'essai peut dire, et ce qu'il ne peut pas :

* il **mesure** l'intelligibilité (la phrase est retranscrite par Whisper et comparée
  à l'original) et le changement de timbre (distance spectrale avec la voix de
  référence) ;
* il **ne peut pas** dire si une voix « sonne bien ». Ça, seule une oreille humaine
  en juge. Le script sert à éliminer les mélanges cassés ou inintelligibles avant
  de les proposer à l'écoute.

    .venv/bin/python tests/bench_voix.py
"""

from __future__ import annotations

import difflib
import os
import unicodedata
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

PHRASE = (
    "Bonjour, ceci est un essai de voix mélangée. "
    "Le chat dort sur le clavier et la lumière du salon est allumée."
)

REFERENCE = "ff_siwis"

# (libellé, spécification de voix)
VOIX = [
    ("ff_siwis (référence)", REFERENCE),
    ("+ ef_dora (es) 50/50", "ff_siwis,ef_dora"),
    ("+ if_sara (it) 50/50", "ff_siwis,if_sara"),
    ("+ af_heart (en) 50/50", "ff_siwis,af_heart"),
    ("+ ef_dora 80/20", "ff_siwis:4+ef_dora:1"),
    ("+ if_sara 80/20", "ff_siwis:4+if_sara:1"),
    ("+ ef_dora 60/40", "ff_siwis:3+ef_dora:2"),
    ("3 voix 60/20/20", "ff_siwis:3+ef_dora:1+if_sara:1"),
]


def nom_fichier(libelle: str) -> str:
    """« + ef_dora (es) 50/50 » → « ef-dora-es-50-50 » (accents retirés)."""
    sans_accent = unicodedata.normalize("NFKD", libelle)
    sans_accent = "".join(c for c in sans_accent if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", sans_accent.lower()).strip("-")


def normaliser(texte: str) -> list[str]:
    texte = texte.lower().replace("’", "'")
    texte = re.sub(r"[^\w'àâäéèêëîïôöùûüç\s]", " ", texte)
    return [m for m in texte.split() if m]


def enveloppe(audio: np.ndarray, n: int = 2048) -> np.ndarray:
    """Spectre moyen : capture grossièrement le timbre (formants)."""
    if audio.size < n:
        return np.zeros(n // 2 + 1)
    fenetres = [
        np.abs(np.fft.rfft(audio[i : i + n] * np.hanning(n)))
        for i in range(0, audio.size - n, n // 2)
    ]
    return np.mean(fenetres, axis=0) if fenetres else np.zeros(n // 2 + 1)


def similarite(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = même timbre, 0.0 = très différent."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main() -> int:
    import soundfile as sf

    from voicechat.stt import Transcriber
    from voicechat.tts import KokoroTTS

    attendu = normaliser(PHRASE)
    dossier = Path(os.environ.get("VOICECHAT_ESSAIS_VOIX", "/tmp/voix"))

    print("=" * 96)
    print("Mélanges de voix Kokoro — intelligibilité mesurée par Whisper")
    print("=" * 96)
    print(f"phrase testée : « {PHRASE} »")
    print()

    tts = KokoroTTS(voice=REFERENCE, lang_code="f", device="auto")
    if not tts.load():
        print(f"TTS indisponible : {tts.load_error}")
        return 2
    print(f"Kokoro chargé sur {tts.device} ({tts.device_reason})")

    stt = Transcriber(modele="small", device="auto", langue="fr")
    if not stt.charger():
        print(f"Whisper indisponible : {stt.erreur}")
        return 2
    print(f"Whisper chargé sur {stt.device} ({stt.compute_reel})")
    print()

    reference_env = None
    resultats: list[tuple[str, float, float, float, float, str]] = []

    for libelle, spec in VOIX:
        t0 = time.monotonic()
        try:
            tts.set_voice(spec)
        except Exception as exc:
            print(f"  {libelle:24} : ÉCHEC au chargement — {exc}")
            continue
        audio = tts.synth(PHRASE)
        duree_synth = time.monotonic() - t0
        if audio.size == 0:
            print(f"  {libelle:24} : audio vide")
            continue

        sf.write("/tmp/bench_voix.wav", audio, 24000)
        # On garde un échantillon de chaque variante : la seule chose que ce script ne
        # peut pas juger, c'est si une voix est agréable. Il faut l'écouter.
        dossier.mkdir(parents=True, exist_ok=True)
        sf.write(str(dossier / f"{nom_fichier(libelle)}.wav"), audio, 24000)
        obtenu = normaliser(stt.transcrire(audio).texte)
        justesse = difflib.SequenceMatcher(None, attendu, obtenu).ratio()
        env = enveloppe(audio)
        if reference_env is None:
            reference_env = env
            distance = 1.0
        else:
            distance = similarite(reference_env, env)
        audio_s = audio.size / 24000
        rtf = duree_synth / audio_s if audio_s else 0.0
        resultats.append((libelle, justesse, len(obtenu) / max(len(attendu), 1),
                          distance, rtf, " ".join(obtenu)))
        print(f"  {libelle:24} : {audio_s:4.2f} s | justesse {justesse:5.1%} | "
              f"timbre vs réf. {distance:.3f} | RTF {rtf:.2f}")

    print()
    print("=" * 96)
    print("Détail des transcriptions")
    print("=" * 96)
    print(f"attendu : {' '.join(attendu)}")
    for libelle, justesse, _, _, _, obtenu in resultats:
        marque = " " if justesse > 0.9 else "!"
        print(f"{marque} {libelle:24} : {obtenu}")

    print()
    print("Lecture : « justesse » = similarité des mots avec l'original (1.0 = parfait).")
    print("« timbre vs réf. » = similarité spectrale avec ff_siwis (1.0 = identique,")
    print("donc mélange inutile ; plus bas = voix réellement différente).")
    print("Le script ne juge PAS si une voix est agréable — ça reste à l'oreille.")
    print()
    print(f"Échantillons écrits dans {dossier} — écoute-les :")
    for fichier in sorted(dossier.glob("*.wav")):
        print(f"  {fichier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
