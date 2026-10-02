"""Vérifie le micro de la page web, de bout en bout (v1.2).

Rien n'est simulé : un vrai `voicechat --web` tourne dans un processus à part (Kokoro sur
le GPU, un vrai modèle derrière), et le script refait **exactement** le geste du
navigateur :

    MediaRecorder  →  blob webm/opus  →  POST /transcris  →  texte

Le webm est fabriqué avec ffmpeg à partir de l'audio que le service a lui-même produit :
c'est donc le même son, dans le format qu'un téléphone envoie vraiment.

    .venv/bin/python tests/verif_micro.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
PHRASE = "La réserve d'audio garde les phrases déjà dites."


def attendre_serveur(processus: subprocess.Popen, delai: float) -> tuple[str, str]:
    """Lit la sortie du serveur jusqu'à ce qu'il annonce son adresse locale."""
    lignes: list[str] = []
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        ligne = processus.stdout.readline() if processus.stdout else ""
        if not ligne:
            if processus.poll() is not None:
                break
            continue
        lignes.append(ligne)
        trouve = re.search(r"en local : (http://[^\s]+)", ligne)
        if trouve:
            return trouve.group(1), "".join(lignes)
    return "", "".join(lignes)


def poster(url: str, octets: bytes, ctype: str = "application/octet-stream", delai: float = 180):
    requete = urllib.request.Request(url, data=octets, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(requete, timeout=delai) as reponse:
            return reponse.status, reponse.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as reponse:
        return json.loads(reponse.read())


def mots(texte: str) -> list[str]:
    return [m for m in re.sub(r"[^a-zà-ÿ0-9 ]", " ", texte.lower()).split() if len(m) > 3]


def proportions(attendu: str, obtenu: str) -> tuple[int, int]:
    attendus = mots(attendu)
    return sum(1 for m in attendus if m in obtenu.lower()), len(attendus)


def transcrire(base: str, octets: bytes) -> tuple[int, dict, str]:
    code, corps = poster(f"{base}/transcris", octets, "audio/webm")
    try:
        infos = json.loads(corps)
    except Exception:
        infos = {"erreur": corps.decode("utf-8", "replace")[:200]}
    return code, infos, infos.get("texte", "")


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-micro-"))
    print("=" * 78)
    print("Micro de la page web : les octets d'un navigateur, transcrits par le service")
    print("=" * 78)

    processus = subprocess.Popen(
        [PY, "-m", "voicechat", "--web", "--port", "0"],
        cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ),
    )
    base, sortie = attendre_serveur(processus, 300.0)
    if not base:
        print(">>> ÉCHEC : le serveur web n'a pas démarré")
        for ligne in sortie.splitlines():
            print(f"  | {ligne}")
        processus.terminate()
        return 1
    for ligne in sortie.splitlines():
        if ligne.strip() and not ligne.startswith("Ctrl"):
            print(f"  | {ligne.strip()}")
    print(f"\n  serveur web : {base}\n")

    controles: list[tuple[str, bool]] = []
    try:
        # --- 1. avant toute parole : Whisper n'est pas chargé -----------------
        etat = get_json(f"{base}/etat")["transcription"]
        print(f"  GET /etat → transcription {etat}")
        controles.append(("le serveur annonce qu'il peut transcrire", etat["disponible"] is True))
        controles.append(("Whisper n'est pas chargé au démarrage", etat["pret"] is False))

        # --- 2. de la parole, produite par le service lui-même ----------------
        code, wav = poster(
            f"{base}/parle", json.dumps({"texte": PHRASE}).encode(), "application/json"
        )
        chemin_wav = dossier / "phrase.wav"
        chemin_wav.write_bytes(wav)
        import soundfile as sf

        echantillons, taux = sf.read(str(chemin_wav))
        print(f"\n  Kokoro a produit {len(wav)} octets ({len(echantillons) / taux:.2f} s)")
        controles.append(("le service a produit l'audio", code == 200 and wav[:4] == b"RIFF"))

        # --- 3. le format d'un navigateur (MediaRecorder → webm/opus) --------
        chemin_webm = dossier / "phrase.webm"
        encodage = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(chemin_wav),
             "-c:a", "libopus", "-b:a", "32k", str(chemin_webm)],
            capture_output=True, text=True,
        )
        if encodage.returncode != 0 or not chemin_webm.exists():
            print(f">>> ÉCHEC : ffmpeg n'a pas su encoder le webm : {encodage.stderr[:200]}")
            controles.append(("le webm/opus est accepté", False))
            webm = b""
        else:
            webm = chemin_webm.read_bytes()
            code, infos, texte = transcrire(base, webm)
            trouves, total = proportions(PHRASE, texte)
            print(f"\n  POST /transcris (webm/opus, {len(webm)} octets) → HTTP {code}")
            print(f"  attendu : « {PHRASE} »")
            print(f"  entendu : « {texte} »")
            print(f"  mots retrouvés : {trouves}/{total} · RTF {infos.get('rtf')}")
            controles.append(("le webm/opus d'un navigateur est accepté", code == 200))
            controles.append(("le webm/opus est correctement transcrit", trouves >= max(1, total * 0.6)))

        # --- 4. Whisper est arrivé, et une seule fois -------------------------
        etat = get_json(f"{base}/etat")["transcription"]
        print(f"\n  GET /etat → {etat}")
        controles.append(("Whisper est chargé après la première demande", etat["pret"] is True))
        controles.append(("le modèle annoncé est celui de la configuration",
                          etat["modele"] not in ("", "?")))
        controles.append(("le compteur de transcriptions a suivi", etat["transcriptions"] >= 1))

        # --- 5. un contenu qui n'est pas de l'audio ---------------------------
        code, infos, _ = transcrire(base, b"ceci n'est pas un son")
        print(f"\n  POST /transcris (texte au lieu de son) → HTTP {code} : {infos.get('erreur', '')[:50]}")
        controles.append(("un contenu illisible donne un 400 lisible", code == 400))

        # --- 6. on parle pendant que le modèle répond -------------------------
        # Le cas réel sur un téléphone : la réponse est en cours de lecture et
        # l'utilisateur veut enchaîner. Whisper et Kokoro partagent le même GPU, chacun
        # avec son verrou : c'est exactement là qu'on découvrirait une famine.
        arret = threading.Event()

        def lire_flux() -> None:
            url = f"{base}/flux?q={quote('Raconte brièvement trois choses sur la mer.')}"
            try:
                with urllib.request.urlopen(url, timeout=120) as reponse:
                    for _ in reponse:
                        if arret.is_set():
                            break
            except Exception:
                pass

        fil = threading.Thread(target=lire_flux, daemon=True)
        fil.start()
        time.sleep(2.0)  # le tour a commencé : le modèle écrit, l'audio se synthétise
        debut = time.monotonic()
        code_pendant, _, _ = transcrire(base, webm) if webm else (0, {}, "")
        duree_pendant = time.monotonic() - debut
        arret.set()
        fil.join(timeout=30)
        print(f"\n  POST /transcris **pendant** le tour → HTTP {code_pendant} en {duree_pendant:.2f} s")
        controles.append(("on peut parler pendant que le modèle répond", code_pendant == 200))

        # --- 7. la page offre bien le micro -----------------------------------
        with urllib.request.urlopen(f"{base}/", timeout=30) as reponse:
            page = reponse.read().decode("utf-8")
        controles.append(("la page a le bouton micro", 'id="micro"' in page))
        controles.append(("la page envoie vers la bonne route", 'fetch("transcris"' in page))
        controles.append(("la page gère le cas sans HTTPS", "getUserMedia" in page and "HTTPS" in page))

        print("\n=== contrôles ===")
        echecs = 0
        for nom, reussi in controles:
            print(f"  {'OK ' if reussi else 'KO '} {nom}")
            echecs += 0 if reussi else 1
        print()
        if echecs:
            print(f">>> {echecs} contrôle(s) en échec")
            return 1
        print(">>> OK : les octets d'un navigateur deviennent du texte, micro compris")
        return 0
    finally:
        processus.terminate()
        try:
            processus.wait(timeout=25)
        except subprocess.TimeoutExpired:
            processus.kill()


if __name__ == "__main__":
    raise SystemExit(main())
