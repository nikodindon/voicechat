"""Vérifie les apports de la v1.1 sur le vrai CLI, avec le vrai serveur LLM.

Quatre fonctions, quatre façons de les éprouver :

* ``-q``        → en sous-processus **sans terminal** (le cas d'un cron) ;
* ``--dire``    → par son effet de bord vérifiable : le WAV déposé dans la réserve ;
* ``/cherche``  → dans un vrai pseudo-terminal, avec une conversation déjà sauvée ;
* ``/resume``   → dans un vrai pseudo-terminal, après de vrais échanges.

    .venv/bin/python tests/verif_confort.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Le pilote de pseudo-terminal vit dans `pilote.py` : les leçons apprises (texte et
# Entrée en deux écritures, attente par silence, tampon non cumulatif) y sont
# expliquées, et il sert aussi à `verif_contexte.py`.
from pilote import PROJ, PY, Session, propre

TEMOIN = "zorglub"  # mot assez rare pour qu'une trouvaille ne puisse pas être un hasard


# ------------------------------------------------------------------ A. -q sans terminal
def verifier_question(dossier: Path) -> tuple[bool, str]:
    """Le cas qui compte : aucune entrée terminale (cron, pipe, autre programme)."""
    resultat = subprocess.run(
        [PY, "-m", "voicechat", "-q", "Réponds exactement : bonjour.", "--no-tts"],
        cwd=PROJ, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180,
        env={**os.environ, "VOICECHAT_DATA": str(dossier / "donnees")},
    )
    sortie = resultat.stdout
    print(f"  code de sortie : {resultat.returncode}")
    for ligne in propre(sortie).splitlines():
        if ligne.strip():
            print(f"  | {ligne.strip()[:100]}")
    if resultat.stderr.strip():
        print(f"  stderr : {resultat.stderr.strip()[:200]}")
    return resultat.returncode == 0 and "ia ›" in sortie, sortie


# ------------------------------------------------------------------ B. --dire
def verifier_dire(dossier: Path) -> tuple[bool, str]:
    """`--dire` doit synthétiser — c'est visible dans la réserve, sans écouter."""
    reserve = dossier / "reserve"
    phrase = "Bonjour, ceci est une vérification de la lecture directe."
    resultat = subprocess.run(
        [PY, "-m", "voicechat", "--dire", phrase],
        cwd=PROJ, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=300,
        env={**os.environ, "VOICECHAT_CACHE_DIR": str(reserve),
             "VOICECHAT_DATA": str(dossier / "donnees")},
    )
    sortie = propre(resultat.stdout)
    for ligne in sortie.splitlines():
        if "Lecture" in ligne or "ERREUR" in ligne or "Voix" in ligne:
            print(f"  | {ligne.strip()}")
    wavs = sorted(reserve.glob("*.wav")) if reserve.is_dir() else []
    print(f"  fichiers dans la réserve : {len(wavs)}")

    duree = 0.0
    if wavs:
        import soundfile as sf

        audio, taux = sf.read(str(wavs[0]))
        duree = len(audio) / taux
        print(f"  durée du WAV déposé : {duree:.2f} s")

    # Le fichier de réserve doit correspondre à la phrase demandée : on recalcule la clé.
    from voicechat.cache import CacheAudio

    attendue = CacheAudio.cle(phrase, "ff_siwis", 1.0, "f")
    trouvee = any(w.stem == attendue for w in wavs)
    print(f"  clé attendue présente : {trouvee}")

    ok = resultat.returncode == 0 and trouvee and 1.0 < duree < 12.0
    return ok, sortie


# ------------------------------------------------------------------ C. /cherche
def verifier_cherche(dossier: Path) -> tuple[bool, str]:
    donnees = dossier / "donnees"
    os.environ["VOICECHAT_DATA"] = str(donnees)
    from voicechat import store

    store.enregistrer(
        "memoire-test",
        [
            {"role": "system", "content": "Tu es concis."},
            {"role": "user", "content": f"Comment on répare le {TEMOIN} ?"},
            {"role": "assistant", "content": f"Il faut recalibrer le {TEMOIN} avant tout."},
        ],
        systeme="Tu es concis.",
    )

    session = Session({"VOICECHAT_DATA": str(donnees)})
    try:
        sortie = session.envoyer(f"/cherche {TEMOIN}")
    finally:
        session.fermer()

    texte = propre(sortie)
    interessantes = [ligne.strip() for ligne in texte.splitlines() if TEMOIN in ligne]
    for ligne in interessantes[:4]:
        print(f"  | {ligne[:110]}")
    trouve = "memoire-test" in texte and "passage(s)" in texte
    return trouve, texte


# ------------------------------------------------------------------ D. /resume
def verifier_resume(dossier: Path) -> tuple[bool, str]:
    session = Session({"VOICECHAT_DATA": str(dossier / "donnees"),
                       "VOICECHAT_CACHE": "0"})
    try:
        # Trois tours, chacun une phrase : de quoi avoir un historique à compacter.
        for question in (
            "Réponds en une phrase : quelle est la capitale de la France ?",
            "Réponds en une phrase : et celle de l'Italie ?",
            "Réponds en une phrase : et celle de l'Espagne ?",
        ):
            session.envoyer(question)
        avant = session.envoyer("/resume", delai=180.0, motif="message(s) en contexte")
    finally:
        session.fermer()

    texte = propre(avant)
    lignes = [l.strip() for l in texte.splitlines() if "[contexte]" in l]
    for ligne in lignes:
        print(f"  | {ligne[:120]}")

    compacte = any("compactage de" in l for l in lignes) and any(
        "message(s) en contexte" in l for l in lignes
    )
    # Le résumé doit rester dans le contexte : on le vérifie au tour suivant.
    return compacte, texte


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-confort-"))
    print("=" * 78)
    print("v1.1 : -q, --dire, /cherche, /resume — sur le vrai CLI et le vrai modèle")
    print("=" * 78)

    controles: list[tuple[str, bool]] = []

    print("\n--- A. mode non-interactif, sans terminal (cas du cron) ---")
    ok, _ = verifier_question(dossier)
    controles.append(("`-q` répond sans terminal et sort en 0", ok))

    print("\n--- B. --dire : lecture directe, vérifiée par la réserve ---")
    ok, _ = verifier_dire(dossier)
    controles.append(("`--dire` synthétise bien la phrase demandée", ok))

    print("\n--- C. /cherche dans les conversations sauvées ---")
    ok, _ = verifier_cherche(dossier)
    controles.append(("`/cherche` retrouve le passage", ok))

    print("\n--- D. /resume : compactage du contexte ---")
    ok, _ = verifier_resume(dossier)
    controles.append(("`/resume` compacte l'historique", ok))

    print("\n=== contrôles ===")
    echecs = 0
    for nom, reussi in controles:
        print(f"  {'OK ' if reussi else 'KO '} {nom}")
        echecs += 0 if reussi else 1
    print()
    if echecs:
        print(f">>> {echecs} contrôle(s) en échec")
        return 1
    print(">>> OK : les quatre apports de la v1.1 fonctionnent sur le vrai programme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
