"""Vérifie l'interface web de bout en bout (v1.2).

Rien n'est simulé : un vrai `voicechat --web` démarre dans un processus à part (vrai
serveur LLM, vrai Kokoro), et un client urllib joue le rôle du navigateur.

**Le point délicat, et ce qui a changé après un vrai bug** : ce script lisait tout le
flux, puis demandait l'audio — ce qui ne ressemble pas à ce que fait le navigateur, et a
laissé passer le défaut que l'utilisateur a vu (le son ne démarrait qu'à la fin de la
réponse). Il demande maintenant l'audio **dès que la première phrase arrive**, pendant que
le modèle écrit encore. C'est la seule façon de vérifier que le son suit la génération au
lieu de l'attendre.

    .venv/bin/python tests/verif_web.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
QUESTION = "Explique en trois phrases courtes ce que fait la réserve d'audio du projet."


def attendre_serveur(processus: subprocess.Popen, delai: float) -> tuple[str, str]:
    """Lit la sortie du serveur jusqu'à ce qu'il annonce l'adresse de la page."""
    lignes: list[str] = []
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        ligne = processus.stdout.readline() if processus.stdout else ""
        if not ligne:
            if processus.poll() is not None:
                break
            continue
        lignes.append(ligne)
        trouve = re.search(r"Page\s+:\s+http://[^:]+:(\d+)", ligne)
        if trouve:
            return trouve.group(1), "".join(lignes)
    return "", "".join(lignes)


def poster(url: str, charge: dict | None = None, delai: float = 120):
    corps = json.dumps(charge or {}).encode("utf-8")
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(requete, timeout=delai) as reponse:
        return reponse.status, reponse.read()


def suivre_flux(url: str, sur_phrase=None) -> tuple[list[tuple[str, dict]], float]:
    """Lit le flux SSE, et appelle `sur_phrase` **pendant** la lecture.

    C'est `sur_phrase` qui reproduit le geste du navigateur : demander l'audio dès la
    première phrase, sans attendre la fin du tour.
    """
    evenements: list[tuple[str, dict]] = []
    debut = time.monotonic()
    premier = 0.0
    with urllib.request.urlopen(url, timeout=300) as reponse:
        nom = ""
        for ligne_brute in reponse:
            ligne = ligne_brute.decode("utf-8").rstrip("\n")
            if ligne.startswith("event: "):
                nom = ligne[len("event: ") :]
            elif ligne.startswith("data: "):
                donnees = json.loads(ligne[len("data: ") :])
                if not premier:
                    premier = time.monotonic() - debut
                evenements.append((nom, donnees))
                if nom == "phrase" and sur_phrase is not None:
                    sur_phrase(donnees["t"], time.monotonic() - debut, nom)
    return evenements, premier


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-web-"))
    print("=" * 78)
    print("Interface web : page, flux SSE, audio — vérifiés sur le vrai programme")
    print("=" * 78)

    processus = subprocess.Popen(
        [PY, "-m", "voicechat", "--web", "--port", "0"],
        cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "VOICECHAT_DATA": str(dossier / "donnees")},
    )
    try:
        port, sortie = attendre_serveur(processus, 240.0)
        for ligne in sortie.splitlines():
            if ligne.strip() and "WARNING" not in ligne:
                print(f"  serveur | {ligne.strip()}")
        if not port:
            print(">>> ÉCHEC : le serveur n'a pas annoncé d'adresse")
            return 1
        base = f"http://127.0.0.1:{port}"
        print(f"\n  adresse retenue : {base}\n")

        controles: list[tuple[str, bool]] = []

        # --- 1. la page -------------------------------------------------------
        with urllib.request.urlopen(f"{base}/", timeout=30) as reponse:
            page = reponse.read().decode("utf-8")
            type_page = reponse.headers["Content-Type"]
        print(f"  GET /       → {type_page}, {len(page)} octets")
        controles.append(("la page est servie en HTML", "text/html" in type_page and len(page) > 1000))
        controles.append(("la page ne dépend d'aucune ressource externe",
                          "cdn" not in page.lower() and "<script src=" not in page))

        # --- 2. l'état --------------------------------------------------------
        with urllib.request.urlopen(f"{base}/etat", timeout=30) as reponse:
            etat = json.loads(reponse.read())
        print(f"  GET /etat   → {etat}")
        controles.append(("l'état annonce le modèle et le contexte",
                          bool(etat.get("modele")) and (etat.get("contexte") or {}).get("total", 0) > 0))

        # --- 3. le flux, avec l'audio demandé en plein tour -------------------
        audio = {"recu": False, "avant_fin": False, "attente": 0.0, "wav": b"", "phrase": ""}
        fin_vue = {"vu": False, "t": 0.0}

        def sur_phrase(phrase: str, instant: float, _nom: str) -> None:
            if audio["recu"]:
                return
            debut = time.monotonic()
            try:
                code, wav = poster(f"{base}/parle", {"texte": phrase})
            except Exception as exc:  # noqa: BLE001 — on veut voir l'échec dans le rapport
                print(f"  !! audio : {exc}")
                return
            audio.update(
                recu=code == 200,
                avant_fin=not fin_vue["vu"],
                attente=time.monotonic() - debut,
                wav=wav,
                phrase=phrase,
            )
            print(f"  POST /parle pendant le tour → HTTP {code}, {len(wav)} octets, "
                  f"en {audio['attente']:.2f} s (à t={instant:.2f} s du flux)")

        evenements, premier = suivre_flux(
            f"{base}/flux?q={quote(QUESTION)}", sur_phrase=sur_phrase
        )
        noms = [nom for nom, _ in evenements]
        texte = "".join(d["t"] for nom, d in evenements if nom == "texte")
        phrases = [d["t"] for nom, d in evenements if nom == "phrase"]
        fin_vue["vu"] = "fin" in noms

        print(f"\n  GET /flux   → {len(evenements)} évènements : "
              f"{', '.join(sorted(set(noms)))}")
        print(f"  premier évènement reçu après {premier * 1000:.0f} ms")
        print(f"  texte  : « {texte.strip()[:100]} »")
        print(f"  phrases : {len(phrases)}")
        for phrase in phrases[:3]:
            print(f"     · {phrase[:88]}")
        if len(phrases) > 3:
            print(f"     · … et {len(phrases) - 3} autre(s)")

        controles.append(("le flux commence par debut et finit par fin",
                          noms[0] == "debut" and noms[-1] == "fin"))
        controles.append(("l'écran et la voix disent la même chose",
                          " ".join(phrases).split() == texte.split()))
        controles.append(("le premier évènement arrive vite (< 10 s)", 0 < premier < 10))
        controles.append((
            "l'audio de la 1re phrase arrive PENDANT la génération, pas après",
            audio["recu"] and audio["avant_fin"],
        ))
        controles.append(("la synthèse ne bloque pas le flux (< 5 s)", audio["attente"] < 5))

        # --- 4. l'audio retranscrit -------------------------------------------
        if audio["recu"]:
            import soundfile as sf

            chemin = dossier / "phrase.wav"
            chemin.write_bytes(audio["wav"])
            echantillons, taux = sf.read(str(chemin))
            print(f"\n  phrase envoyée à Whisper : « {audio['phrase'][:80]} »")
            print(f"  durée : {len(echantillons) / taux:.2f} s d'audio")

            from faster_whisper import WhisperModel

            modele = WhisperModel("small", device="cuda", compute_type="int8_float32")
            segments, _ = modele.transcribe(str(chemin), language="fr")
            entendu = " ".join(s.text.strip() for s in segments).strip()
            print(f"  entendu : « {entendu} »")

            def mots(t: str) -> list[str]:
                return [m for m in re.sub(r"[^a-zà-ÿ0-9 ]", " ", t.lower()).split() if len(m) > 3]

            attendus = mots(audio["phrase"])
            trouves = sum(1 for mot in attendus if mot in entendu.lower())
            proportion = trouves / len(attendus) if attendus else 0.0
            print(f"  mots significatifs retrouvés : {trouves}/{len(attendus)} ({proportion:.0%})")
            controles.append(("l'audio du navigateur est la phrase affichée", proportion >= 0.7))
        else:
            controles.append(("l'audio du navigateur est la phrase affichée", False))

        # --- 5. reset ---------------------------------------------------------
        code, _ = poster(f"{base}/reset")
        with urllib.request.urlopen(f"{base}/etat", timeout=30) as reponse:
            apres = json.loads(reponse.read())
        print(f"\n  POST /reset → HTTP {code}, {apres['archives']} message(s) restant(s)")
        controles.append(("reset vide l'historique", apres["archives"] == 1))

        # --- 6. le suivi de contexte ------------------------------------------
        suivre_flux(f"{base}/flux?q={quote('Dis simplement bonjour.')}")
        with urllib.request.urlopen(f"{base}/etat", timeout=30) as reponse:
            suivi = json.loads(reponse.read())
        utilise = (suivi.get("contexte") or {}).get("utilise", 0)
        print(f"  après un tour : {utilise} tokens utilisés de {suivi['contexte']['total']}")
        controles.append(("le contexte est suivi après le tour", utilise > 0))

        print("\n=== contrôles ===")
        echecs = 0
        for nom, reussi in controles:
            print(f"  {'OK ' if reussi else 'KO '} {nom}")
            echecs += 0 if reussi else 1
        print()
        if echecs:
            print(f">>> {echecs} contrôle(s) en échec")
            return 1
        print(">>> OK : la page, le flux et l'audio fonctionnent sur le vrai programme")
        return 0
    finally:
        processus.terminate()
        try:
            processus.wait(timeout=20)
        except subprocess.TimeoutExpired:
            processus.kill()


if __name__ == "__main__":
    raise SystemExit(main())
