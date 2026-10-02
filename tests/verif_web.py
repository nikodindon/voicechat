"""Vérifie l'interface web de bout en bout (v1.2).

Rien n'est simulé : un vrai `voicechat --web` démarre dans un processus à part (vrai
serveur LLM, vrai Kokoro), et un client urllib joue le rôle du navigateur.

Ce qui est éprouvé, dans l'ordre :

* la page est servie, et elle ne dépend d'aucune ressource externe ;
* l'état annonce le bon modèle et la bonne taille de contexte ;
* le flux SSE rend le texte **et** les phrases à dire, et les deux se recollent ;
* l'audio récupéré par HTTP sur la première phrase est **retranscrit par Whisper** —
  c'est le seul contrôle qui prouve que le navigateur jouerait bien la phrase affichée ;
* `/reset` vide l'historique côté serveur.

    .venv/bin/python tests/verif_web.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
QUESTION = "Réponds en une phrase simple : que fais-tu quand le GPU est occupé ?"


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


def lire_flux(url: str) -> tuple[list[tuple[str, dict]], float]:
    """Lit le flux SSE, horodate le premier évènement, rend les évènements."""
    evenements: list[tuple[str, dict]] = []
    debut = time.monotonic()
    premier = 0.0
    with urllib.request.urlopen(url, timeout=180) as reponse:
        nom = ""
        for ligne_brute in reponse:
            ligne = ligne_brute.decode("utf-8").rstrip("\n")
            if ligne.startswith("event: "):
                nom = ligne[len("event: ") :]
            elif ligne.startswith("data: "):
                if not premier:
                    premier = time.monotonic() - debut
                evenements.append((nom, json.loads(ligne[len("data: ") :])))
    return evenements, premier


def poster(url: str, charge: dict | None = None):
    corps = json.dumps(charge or {}).encode("utf-8")
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(requete, timeout=180) as reponse:
        return reponse.status, reponse.read()


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-web-"))
    print("=" * 78)
    print("Interface web : page, flux SSE, audio — vérifiés sur le vrai programme")
    print("=" * 78)

    processus = subprocess.Popen(
        [PY, "-m", "voicechat", "--web", "--port", "0"],
        cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**_environ(), "VOICECHAT_DATA": str(dossier / "donnees")},
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

        # --- 3. le flux -------------------------------------------------------
        evenements, premier = lire_flux(f"{base}/flux?q={quote(QUESTION)}")
        noms = [nom for nom, _ in evenements]
        texte = "".join(d["t"] for nom, d in evenements if nom == "texte")
        phrases = [d["t"] for nom, d in evenements if nom == "phrase"]
        fin = dict(evenements).get("fin", {})
        print(f"\n  GET /flux   → {len(evenements)} évènements : "
              f"{', '.join(sorted(set(noms)))}")
        print(f"  premier évènement reçu après {premier * 1000:.0f} ms")
        print(f"  texte  : « {texte.strip()[:110]} »")
        print(f"  phrases : {len(phrases)}")
        for phrase in phrases:
            print(f"     · {phrase[:90]}")

        controles.append(("le flux commence par debut et finit par fin",
                          noms[0] == "debut" and noms[-1] == "fin"))
        controles.append(("l'écran et la voix disent la même chose",
                          " ".join(phrases).split() == texte.split()))
        controles.append(("le premier évènement arrive vite (< 10 s)", 0 < premier < 10))
        controles.append(("le texte est arrivé avant la fin du tour",
                          noms.count("texte") > 1 and "fin" in noms))

        # --- 4. l'audio par HTTP, retranscrit ---------------------------------
        if phrases:
            import soundfile as sf

            code, wav = poster(f"{base}/parle", {"texte": phrases[0]})
            chemin = dossier / "phrase.wav"
            chemin.write_bytes(wav)
            audio, taux = sf.read(str(chemin))
            duree = len(audio) / taux
            print(f"\n  POST /parle → HTTP {code}, {len(wav)} octets, {duree:.2f} s d'audio")

            from faster_whisper import WhisperModel

            modele = WhisperModel("small", device="cuda", compute_type="int8_float32")
            segments, _ = modele.transcribe(str(chemin), language="fr")
            entendu = " ".join(s.text.strip() for s in segments).strip()
            print(f"  entendu : « {entendu} »")

            def mots(t: str) -> list[str]:
                return [m for m in re.sub(r"[^a-zà-ÿ0-9 ]", " ", t.lower()).split() if len(m) > 3]

            attendus = mots(phrases[0])
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
        lire_flux(f"{base}/flux?q={quote('Dis simplement bonjour.')}")
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


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
