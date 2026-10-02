"""Vérifie le mode distribué de bout en bout (v1.0).

Rien n'est simulé : un vrai `voicechat --serveur-tts` démarre dans un processus à
part (Kokoro réel, GPU réel), un vrai `TTSDistant` lui envoie du texte par HTTP, et
l'audio qui revient est **retranscrit par Whisper** pour prouver que c'est bien la
phrase demandée — un WAV valide mais muet passerait tous les autres contrôles.

    .venv/bin/python tests/verif_distant.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
PHRASE = "Le serveur de synthèse répond depuis un autre processus."


def attendre_serveur(processus: subprocess.Popen, delai: float) -> tuple[str, str]:
    """Lit la sortie du serveur jusqu'à ce qu'il annonce son adresse."""
    lignes: list[str] = []
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        ligne = processus.stdout.readline() if processus.stdout else ""
        if not ligne:
            if processus.poll() is not None:
                break
            continue
        lignes.append(ligne)
        trouve = re.search(r"Écoute\s+:\s+http://[^:]+:(\d+)", ligne)
        if trouve:
            return trouve.group(1), "".join(lignes)
    return "", "".join(lignes)


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-distant-"))
    print("=" * 78)
    print("Mode distribué : serveur TTS réel + client réel, audio vérifié par Whisper")
    print("=" * 78)

    processus = subprocess.Popen(
        [PY, "-m", "voicechat", "--serveur-tts", "--port", "0"],
        cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        port, sortie = attendre_serveur(processus, 180.0)
        for ligne in sortie.splitlines():
            if ligne.strip() and "WARNING" not in ligne:
                print(f"  serveur | {ligne.strip()}")
        if not port:
            print(">>> ÉCHEC : le serveur n'a pas annoncé d'adresse")
            return 1

        url = f"http://127.0.0.1:{port}"
        print(f"\n  adresse retenue pour le test : {url}\n")

        with urllib.request.urlopen(f"{url}/sante", timeout=15) as reponse:
            print(f"  GET /sante → {reponse.status} {reponse.read().decode()}")

        from voicechat.distant import TTSDistant

        client = TTSDistant(url=url, voice="ff_siwis")
        if not client.load():
            print(f">>> ÉCHEC : {client.load_error}")
            return 1
        print(f"  client chargé : {client.device_reason}\n")

        # --- premier appel : synthèse réelle ---------------------------------
        t0 = time.monotonic()
        audio = client.synth(PHRASE)
        froid = time.monotonic() - t0
        duree = audio.size / 24000
        print(f"  appel 1 (froid)  : {audio.size} échantillons = {duree:.2f} s d'audio "
              f"en {froid:.2f} s (RTF {froid / duree:.2f})")

        # --- deuxième appel : la phrase est déjà dans la réserve -------------
        t0 = time.monotonic()
        audio2 = client.synth(PHRASE)
        chaud = time.monotonic() - t0
        print(f"  appel 2 (réserve): {audio2.size} échantillons en {chaud:.3f} s")

        chemin = dossier / "repris.wav"
        import soundfile as sf

        sf.write(str(chemin), audio, 24000, subtype="FLOAT")
        print(f"  audio écrit      : {chemin}")

        # --- preuve : Whisper doit reconnaître la phrase ---------------------
        print("\n  retranscription de l'audio revenu par le réseau (Whisper small)…")
        from faster_whisper import WhisperModel

        modele = WhisperModel("small", device="cuda", compute_type="int8_float32")
        segments, _ = modele.transcribe(str(chemin), language="fr")
        texte = " ".join(s.text.strip() for s in segments).strip()
        print(f"  entendu : « {texte} »")

        def normaliser(t: str) -> str:
            return re.sub(r"[^a-zà-ÿ0-9 ]", "", t.lower())

        attendu = normaliser(PHRASE)
        obtenu = normaliser(texte)
        # Whisper écrit parfois « synthèse » en plusieurs mots : on compare les mots
        # communs plutôt qu'une égalité stricte, qui échouerait sur une virgule.
        mots_attendus = [m for m in attendu.split() if len(m) > 3]
        trouves = sum(1 for m in mots_attendus if m in obtenu)
        proportion = trouves / len(mots_attendus) if mots_attendus else 0.0
        print(f"  mots significatifs retrouvés : {trouves}/{len(mots_attendus)} "
              f"({proportion:.0%})")

        controles = [
            ("le serveur a annoncé son adresse", bool(port)),
            ("GET /sante répond 200", True),
            ("le client s'est chargé", client.ready),
            ("l'audio a la bonne taille", audio.size > 24000),
            ("le second appel est plus rapide", chaud < froid),
            ("Whisper reconnaît la phrase", proportion >= 0.7),
        ]
        print("\n=== contrôles ===")
        echecs = 0
        for nom, ok in controles:
            print(f"  {'OK ' if ok else 'KO '} {nom}")
            echecs += 0 if ok else 1
        print()
        if echecs:
            print(f">>> {echecs} contrôle(s) en échec")
            return 1
        print(">>> OK : texte → HTTP → GPU → WAV → Whisper, la boucle est bouclée")
        return 0
    finally:
        processus.terminate()
        try:
            processus.wait(timeout=15)
        except subprocess.TimeoutExpired:
            processus.kill()


if __name__ == "__main__":
    raise SystemExit(main())
