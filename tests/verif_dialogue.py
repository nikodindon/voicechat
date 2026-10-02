"""Vérifie le mode dialogue : deux personas, deux voix, une alternance (v1.5).

Rien n'est simulé côté modèles : un vrai Kokoro sur le GPU, le vrai serveur LLM, et les
vrais profils. Une seule pièce est remplacée — le haut-parleur — et **pas pour aller plus
vite** : il encaisse l'audio au lieu de le jouer, mais il prend le même temps qu'un vrai.

C'est ce qui rend la vérification utile. Le défaut recherché est celui-ci :

    ``SpeechPipeline`` synthétise dans un thread et lit ``tts.voice`` au **dépilement**.
    Si on bascule la voix sans avoir vidé la file, les dernières phrases d'une persona
    sont dites avec la voix de la suivante.

Avec un haut-parleur instantané, il ne resterait jamais rien en attente et le défaut
resterait invisible. En lisant en temps réel, la file contient toujours quelque chose au
moment de la bascule — exactement comme dans la vraie vie.

    .venv/bin/python tests/verif_dialogue.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUJET = "Faut-il préférer une réponse courte et sûre, ou longue et incertaine ?"
ALICE = "Tu es Alice, tu parles en deux phrases courtes, avec des idées nettes et tranchées."
BOB = "Tu es Bob, tu parles en deux phrases courtes, et tu donnes toujours un exemple concret."


class EnceinteFactice:
    """Encaisse l'audio au lieu de le jouer — mais prend le temps d'un vrai haut-parleur.

    Un `Speaker` instantané rendrait la vérification aveugle : voir l'en-tête du fichier.
    """

    def __init__(self, samplerate: int = 24000) -> None:
        self.samplerate = samplerate
        self.tampons: list = []
        self.secondes = 0.0

    @property
    def available(self) -> bool:
        return True

    @property
    def pret(self) -> bool:
        return True

    def start(self) -> None:
        pass

    def play(self, audio) -> None:
        import numpy as np

        bloc = np.asarray(audio)
        self.tampons.append(bloc)
        duree = len(bloc) / self.samplerate
        self.secondes += duree
        time.sleep(duree)  # c'est le cœur de la vérification

    def wait(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-dialogue-"))
    os.environ["VOICECHAT_PROFILS"] = str(dossier)

    from voicechat import store
    from voicechat.cli import ChatSession
    from voicechat.config import Config
    from voicechat.dialogue import Dialogue, Persona
    from voicechat.tts import decrire_voix

    print("=" * 78)
    print("Mode dialogue : deux personas, deux voix, une alternance")
    print("=" * 78)

    # 1. Deux personnes, deux profils, deux voix ------------------------------------
    store.enregistrer_profil("alice", ALICE, voix="ff_siwis", vitesse=1.05)
    store.enregistrer_profil("bob", BOB, voix="ef_dora")
    print(f"\n[1/5] profils écrits dans {dossier}")
    for nom in ("alice", "bob"):
        chemin = dossier / f"{nom}.md"
        for ligne in chemin.read_text(encoding="utf-8").splitlines():
            if ligne.strip():
                print(f"      {nom} | {ligne}")

    # 2. La session : vrai Kokoro, faux haut-parleur --------------------------------
    session = ChatSession(Config(tts=True, cache=True))
    if not session.resolve_model():
        print(">>> ÉCHEC : aucun modèle joignable sur le serveur")
        return 2
    session.cfg, _ = session.cfg.pour_modele(session.cfg.model)
    print(f"\n[2/5] modèle : {session.cfg.model}")
    session.setup_voice()
    if session.tts is None:
        print(f">>> ÉCHEC : Kokoro indisponible ({session.cfg.tts})")
        return 2
    if session.speaker:
        session.speaker.close()
    if session.speech:
        session.speech.close()
    enceinte = EnceinteFactice()
    session.speaker = enceinte  # type: ignore[assignment]
    session._brancher_pipeline()
    # Référence locale non optionnelle : dans les fermetures ci-dessous, Pyright ne saurait
    # pas que `session.tts` a été vérifié juste avant.
    tts = session.tts
    print(f"\n[2/5] Kokoro : {decrire_voix(tts.voice)} sur {tts.device}")
    print(f"      haut-parleur : factice (le son est capturé, pas joué)")

    # Les personas sont chargés ici : l'instrumentation a besoin de leurs voix attendues.
    a = Persona.charger("alice")
    b = Persona.charger("bob")
    attendues = {persona.nom: persona.voix for persona in (a, b)}
    print(f"      voix attendues, d'après les profils : {attendues}")

    # 3. L'instrumentation ----------------------------------------------------------
    # On note la voix **au moment de chaque synthèse** : c'est la seule mesure qui dise
    # la vérité, parce que c'est là que le thread de synthèse lit `tts.voice`.
    #
    # Et on compare à la voix que le **profil demande** pour ce persona — pas à celle qui
    # se trouve chargée. Comparer à ce qui est en place rend le contrôle aveugle : si on
    # oublie d'appliquer la voix, les deux côtés disent la même chose et il passe au vert.
    syntheses: list[tuple[str, str]] = []
    desynchro: list[tuple[str, str, str]] = []
    attendu = {"voix": ""}

    synth_reel = tts.synth

    def synth_suivi(texte: str):
        syntheses.append((tts.voice, texte))
        if attendu["voix"] and tts.voice != attendu["voix"]:
            desynchro.append((texte[:40], tts.voice, attendu["voix"]))
        return synth_reel(texte)

    tts.synth = synth_suivi  # type: ignore[method-assign]

    tour_reel = session.tour_modele

    def tour_suivi(messages, etiquette="ia › ", modele=None):
        # À qui est ce tour ? L'étiquette porte le nom du persona ; la voix attendue vient
        # de son profil, lu comme le fait la console.
        nom = etiquette.split(" › ")[0]
        attendu["voix"] = attendues.get(nom, "")
        return tour_reel(messages, etiquette, modele)

    session.tour_modele = tour_suivi  # type: ignore[method-assign]

    # 4. Le dialogue ----------------------------------------------------------------
    dialogue = Dialogue(session, a, b, SUJET, tours=4)
    print()
    dialogue.annoncer()
    print("─" * 78)
    debut = time.monotonic()
    repliques = dialogue.derouler()
    duree = time.monotonic() - debut
    print("─" * 78)

    # 5. Les contrôles --------------------------------------------------------------
    controles: list[tuple[str, bool]] = []

    controles.append(("le dialogue a produit ses 4 répliques", len(repliques) == 4))
    noms = [nom for nom, _ in repliques]
    controles.append(("les deux personas alternent", noms == ["alice", "bob", "alice", "bob"]))
    controles.append(("aucune réplique vide", all(t.strip() for _, t in repliques)))

    # Les trois dérives de texte observées à l'usage, et qu'aucun contrôle de voix ne
    # pouvait voir. Le modèle les produit tout seul si on ne l'en empêche pas.
    import re

    textes = [t for _, t in repliques]
    prefixes = [
        (nom, t[:30])
        for nom, t in repliques
        if re.match(r"^\s*[*_#]{0,3}\s*(alice|bob)\s*:", t, re.I)
    ]
    if prefixes:
        print("\n  !! répliques qui s'ouvrent par un nom :")
        for nom, debut in prefixes[:4]:
            print(f"     {nom} › {debut}…")
    controles.append(("aucune réplique ne s'ouvre par un nom", not prefixes))

    recopies = []
    for i in range(1, len(textes)):
        debut = textes[i - 1][:40].strip().lower()
        if len(debut) > 20 and textes[i].strip().lower().startswith(debut):
            recopies.append(noms[i])
    if recopies:
        print(f"\n  !! répliques qui recopient le message reçu : {recopies}")
    controles.append(("aucune réplique ne recopie le message reçu", not recopies))

    plus_longue = max((len(t.split()) for t in textes), default=0)
    print(f"  réplique la plus longue : {plus_longue} mots (limite {dialogue.mots_max})")
    controles.append(
        (
            f"aucune réplique ne part en pavé (max {dialogue.mots_max} mots)",
            plus_longue <= dialogue.mots_max,
        )
    )

    # C'est le contrôle central : aucune phrase ne doit être synthétisée avec la voix
    # d'un autre persona que celui dont c'était le tour.
    if desynchro:
        print("\n  !! phrases dites avec la mauvaise voix :")
        for phrase, voix, attendue in desynchro[:5]:
            print(f"     « {phrase}… » — voix {voix}, attendue {attendue}")
    controles.append(
        ("chaque phrase est dite par la voix du persona dont c'était le tour", not desynchro)
    )

    voix_utilisees = sorted({voix for voix, _ in syntheses})
    print(f"\n  voix qui ont réellement synthétisé : {voix_utilisees}")
    print(f"  phrases synthétisées : {len(syntheses)}")
    controles.append(("les deux voix ont réellement parlé", len(voix_utilisees) == 2))

    # La voix de la session ne doit pas rester celle d'un personnage — vérifié **avant**
    # le test témoin ci-dessous, qui change la voix exprès.
    controles.append(("la session a retrouvé sa voix", session.cfg.voice == tts.voice))

    # La preuve que les deux voix sont bien distinctes : la même phrase, deux voix.
    phrase = "Voici une phrase témoin, dite par deux voix différentes."
    tts.set_voice("ff_siwis")
    temoin_a = tts.synth(phrase)
    tts.set_voice("ef_dora")
    temoin_b = tts.synth(phrase)
    import numpy as np

    n = min(len(temoin_a), len(temoin_b))
    correlation = float(np.corrcoef(temoin_a[:n], temoin_b[:n])[0, 1]) if n else 0.0
    print(f"  même phrase, deux voix : corrélation {correlation:.3f} "
          f"(1.000 = identiques)")
    controles.append(("les deux voix donnent vraiment deux sons", abs(correlation) < 0.9))

    print(f"\n  audio produit : {enceinte.secondes:.1f} s en {duree:.1f} s "
          f"({len(enceinte.tampons)} tampons)")
    print(f"  RTF de synthèse : {tts.rtf:.2f}")

    print("\n=== contrôles ===")
    echecs = 0
    for nom, reussi in controles:
        print(f"  {'OK ' if reussi else 'KO '} {nom}")
        echecs += 0 if reussi else 1

    if session.speech:
        session.speech.close()
    print()
    if echecs:
        print(f">>> {echecs} contrôle(s) en échec")
        return 1
    print(">>> OK : deux personas, deux voix — et chacune dit bien ses propres répliques")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
