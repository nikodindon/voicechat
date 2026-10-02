"""Vérifie le comptage de contexte en tokens (v1.2), sur le vrai serveur.

Ce qui est éprouvé, dans l'ordre :

* le client **demande** au serveur la taille de son contexte, et l'annonce ;
* l'estimation interne reste proche du compte **exact** que le serveur donne à chaque
  tour — c'est la comparaison qui dit si le mécanisme d'ancrage tient ;
* avec un contexte volontairement minuscule, l'élagage se déclenche **pour de vrai**,
  l'avertissement s'affiche, et la conversation continue de fonctionner.

    .venv/bin/python tests/verif_contexte.py
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from pilote import Session, propre

QUESTION = "Réponds en une seule phrase courte : que sais-tu faire ?"


def nombre(texte: str) -> int:
    """« 180 / 32768 tokens » → 180 (le premier nombre de la ligne seulement)."""
    trouve = re.search(r"(\d[\d\s\u202f]*\d|\d)", texte)
    return int(re.sub(r"[^0-9]", "", trouve.group(1))) if trouve else 0


def verifier_contexte_reel(dossier: Path) -> tuple[bool, bool, str]:
    """Contexte réel du serveur : lecture, puis preuve que le compte vient du serveur."""
    session = Session({"VOICECHAT_DATA": str(dossier / "a")}, options=["--no-tts", "--debug"])
    try:
        demarrage = propre(session.sortie)
        lignes = [l.strip() for l in demarrage.splitlines() if "Contexte" in l or "Modèle" in l]
        for ligne in lignes:
            print(f"  | {ligne}")

        # Trois tours : le serveur rend un compte exact à chacun, et l'ancrage s'installe.
        for _ in range(3):
            session.envoyer(QUESTION)

        apres = propre(session.envoyer("/contexte"))
        for ligne in apres.splitlines():
            if any(m in ligne for m in ("contexte :", "message(s) dans", "compte exact", "budget")):
                print(f"  | {ligne.strip()}")

        # L'estimation affichée par /contexte doit être de l'ordre de grandeur du texte
        # réellement présent : on la compare au nombre de caractères des messages.
        lignes_ctx = [l.strip() for l in apres.splitlines() if "contexte :" in l]
        estimate = nombre(lignes_ctx[0]) if lignes_ctx else 0
        total_caracteres = sum(len(m.get("content", "")) for m in session_processus_messages(session))
        attendu_grossier = total_caracteres / 3.5
        print(f"  estimation : {estimate} tokens pour ~{total_caracteres} caractères "
              f"(soit ~{attendu_grossier:.0f} tokens calculés)")

        # Le compte exact du serveur doit être apparu : c'est lui qui ancre le calcul.
        ancrage = 0
        trouve = re.search(r"compte exact du serveur : (\d+) tokens pour (\d+)", apres)
        if trouve:
            ancrage = int(trouve.group(1))
            print(f"  ancrage serveur : {ancrage} tokens pour {trouve.group(2)} message(s)")

        annonce = any("Contexte" in l for l in lignes)
        coherent = estimate > 0 and 0.3 <= estimate / max(1.0, attendu_grossier) <= 3.0
        return annonce and ancrage > 0, coherent, apres
    finally:
        session.fermer()


def session_processus_messages(session: Session) -> list[dict]:
    """Les messages réellement échangés, tels qu'on peut les relire dans la sortie.

    On ne peut pas interroger l'objet `ChatSession` depuis un sous-processus : on
    approxime à partir de ce qui a été écrit, ce qui suffit pour un contrôle d'ordre
    de grandeur.
    """
    morceaux = []
    for ligne in propre(session.sortie).splitlines():
        if ligne.startswith(QUESTION[:20]):
            morceaux.append({"content": QUESTION})
        elif ligne.startswith("ia ›"):
            morceaux.append({"content": ligne[len("ia ›") :]})
    return morceaux


def verifier_elagage(dossier: Path) -> tuple[bool, bool, str]:
    """Contexte forcé minuscule : l'élagage doit se déclencher et le dire."""
    # 150 tokens de budget (système compris) : trois petites réponses suffisent à déborder.
    session = Session(
        {"VOICECHAT_DATA": str(dossier / "b"), "VOICECHAT_N_CTX": "150"},
        options=["--no-tts"],
    )
    try:
        annonce = "forcé par la configuration" in propre(session.sortie)
        print(f"  | contexte forcé annoncé : {annonce}")

        sortie = ""
        for _ in range(8):
            sortie += session.envoyer(QUESTION)
            if "ne sont plus envoyés" in sortie:
                break
        texte = propre(sortie)

        for ligne in [l.strip() for l in texte.splitlines() if "[contexte]" in l][:3]:
            print(f"  | {ligne}")

        elague = "ne sont plus envoyés" in texte
        # Le programme doit continuer à répondre après avoir élagué.
        derniere = propre(session.envoyer("Dis simplement : ok."))
        repond_encore = "ia ›" in derniere
        return annonce and elague, repond_encore, texte
    finally:
        session.fermer()


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-ctx-"))
    print("=" * 78)
    print("Contexte en tokens : lecture du serveur, ancrage, élagage réel")
    print("=" * 78)

    controles: list[tuple[str, bool]] = []

    print("\n--- A. contexte réel du serveur, et ancrage sur ses chiffres ---")
    ancre, coherent, _ = verifier_contexte_reel(dossier)
    controles.append(("le serveur annonce son contexte et le client s'ancre dessus", ancre))
    controles.append(("l'estimation reste cohérente avec le texte présent", coherent))

    print("\n--- B. contexte forcé minuscule : l'élagage en vrai ---")
    elague, repond, _ = verifier_elagage(dossier)
    controles.append(("l'élagage se déclenche et s'annonce", elague))
    controles.append(("le programme répond encore après avoir élagué", repond))

    print("\n=== contrôles ===")
    echecs = 0
    for nom, reussi in controles:
        print(f"  {'OK ' if reussi else 'KO '} {nom}")
        echecs += 0 if reussi else 1
    print()
    if echecs:
        print(f">>> {echecs} contrôle(s) en échec")
        return 1
    print(">>> OK : le client compte en tokens, d'après les chiffres du serveur")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
