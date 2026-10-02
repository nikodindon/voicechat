"""Vérifie qu'un profil porte bien sa voix : persona → voix + vitesse + langue.

Le point de la v0.5 : `/profil <nom>` ne doit pas seulement changer le prompt
système, mais aussi la voix associée — sinon chaque persona se ressemble.

On exerce le **vrai** chemin de code (`ChatSession`, qui applique le profil dans
son constructeur, puis `setup_voice`) avec un vrai Kokoro, dans un dossier de
profils isolé pour ne pas toucher à ceux de l'utilisateur.

    .venv/bin/python tests/verif_profil_voix.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = "Tu es un narrateur posé, tu réponds en français."


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-profils-"))
    os.environ["VOICECHAT_PROFILS"] = str(dossier)
    os.environ.setdefault(
        "PULSE_SOURCE", "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
    )

    from voicechat import store
    from voicechat.config import Config
    from voicechat.cli import ChatSession
    from voicechat.tts import decrire_voix

    print("=" * 78)
    print("Un profil porte sa voix (persona → voix)")
    print("=" * 78)

    # 1. On écrit un profil exactement comme le ferait /profil save
    chemin = store.enregistrer_profil(
        "narrateur", PROMPT, voix="ff_siwis:3+ef_dora:1", vitesse=1.1
    )
    print(f"\n[1/4] profil écrit : {chemin}")
    print("      contenu :")
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        print(f"        | {ligne}")

    # 2. Le session lit le profil au démarrage (--profil)
    session = ChatSession(Config(profil="narrateur"))
    print("\n[2/4] profil appliqué par ChatSession :")
    print(f"      prompt  : {session.cfg.system[:50]!r}…")
    print(f"      voix    : {session.cfg.voice}")
    print(f"      vitesse : {session.cfg.speed}")
    print(f"      message système = prompt du profil : "
          f"{session.messages[0]['content'] == PROMPT}")

    ok = True
    for nom, obtenu, attendu in (
        ("prompt système", session.cfg.system, PROMPT),
        ("voix", session.cfg.voice, "ff_siwis:3+ef_dora:1"),
        ("vitesse", session.cfg.speed, 1.1),
    ):
        bon = obtenu == attendu
        ok &= bon
        print(f"      {'OK ' if bon else 'KO '} {nom}")

    # 3. Le TTS charge bien le mélange
    print("\n[3/4] chargement de Kokoro avec le mélange…")
    session.setup_voice()
    if not session.tts:
        print(f"      TTS indisponible — {session.cfg.tts}")
        return 2
    print(f"      voix chargée : {decrire_voix(session.tts.voice)}")
    print(f"      vitesse TTS  : {session.tts.speed}")
    ok &= session.tts.voice == "ff_siwis:3+ef_dora:1"

    # 4. Et il synthétise vraiment avec
    print("\n[4/4] synthèse réelle avec la voix du profil :")
    audio = session.tts.synth("Le narrateur entre en scène, et la lumière baisse.")
    duree = len(audio) / 24000 if audio.size else 0.0
    print(f"      {len(audio)} échantillons = {duree:.2f} s d'audio")
    ok &= duree > 0.5

    session.ecouteur and session.ecouteur.micro.fermer()
    if session.speaker:
        session.speaker.close()

    print()
    if ok:
        print(">>> OK : le profil a porté sa voix jusqu'à la synthèse")
        return 0
    print(">>> ÉCHEC : au moins un contrôle est tombé")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
