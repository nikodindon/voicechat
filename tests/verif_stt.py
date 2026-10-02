"""Vérifie le service de transcription de bout en bout (v1.2).

Rien n'est simulé : deux vrais services tournent dans des processus à part — le serveur
TTS (Kokoro, GPU) et le serveur STT (Whisper, GPU) — et le script fait passer de la
parole par les deux, dans les deux sens.

C'est la **boucle complète** qui est éprouvée :

    texte  →  Kokoro (service TTS)  →  WAV  →  Whisper (service STT)  →  texte

Plus trois formats d'entrée, parce que c'est là qu'on se trompe :

* WAV 24 kHz mono — ce que produit Kokoro ;
* **webm/opus**, exactement ce que produit le micro d'un navigateur (`MediaRecorder`) et
  que libsndfile ne sait pas lire : c'est le repli PyAV qui doit prendre le relais ;
* WAV 44,1 kHz **stéréo** — ce que produit un micro de machine.

    .venv/bin/python tests/verif_stt.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
PHRASE = "La réserve d'audio garde les phrases déjà dites."


def demarrer(commande: list[str], marqueur: str, delai: float) -> tuple[str, str, subprocess.Popen]:
    """Lance un service et attend qu'il annonce son adresse."""
    processus = subprocess.Popen(
        [PY, "-m", "voicechat", *commande],
        cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ),
    )
    lignes: list[str] = []
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        ligne = processus.stdout.readline() if processus.stdout else ""
        if not ligne:
            if processus.poll() is not None:
                break
            continue
        lignes.append(ligne)
        trouve = re.search(rf"{marqueur}\s+:\s+http://[^:]+:(\d+)", ligne)
        if trouve:
            return trouve.group(1), "".join(lignes), processus
    return "", "".join(lignes), processus


def poster(url: str, donnees: bytes, ctype: str, delai: float = 180):
    requete = urllib.request.Request(
        url, data=donnees, headers={"Content-Type": ctype}, method="POST"
    )
    try:
        with urllib.request.urlopen(requete, timeout=delai) as reponse:
            return reponse.status, reponse.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def mots(texte: str) -> list[str]:
    """Mots signifiants, pour comparer deux transcriptions sans pinailler."""
    return [m for m in re.sub(r"[^a-zà-ÿ0-9 ]", " ", texte.lower()).split() if len(m) > 3]


def proportions(attendu: str, obtenu: str) -> tuple[int, int]:
    attendus = mots(attendu)
    trouves = sum(1 for mot in attendus if mot in obtenu.lower())
    return trouves, len(attendus)


def transcrire(base_stt: str, octets: bytes) -> tuple[int, dict, str]:
    code, corps = poster(f"{base_stt}/transcris", octets, "application/octet-stream")
    try:
        infos = json.loads(corps)
    except Exception:
        infos = {"erreur": corps.decode("utf-8", "replace")[:200]}
    return code, infos, infos.get("texte", "")


def main() -> int:
    dossier = Path(tempfile.mkdtemp(prefix="vc-stt-"))
    print("=" * 78)
    print("Service de transcription : boucle texte → audio → texte, sur deux services")
    print("=" * 78)

    port_tts, sortie_tts, tts = demarrer(["--serveur-tts", "--port", "0"], "Écoute", 240.0)
    if not port_tts:
        print(">>> ÉCHEC : le serveur TTS n'a pas démarré")
        for ligne in sortie_tts.splitlines():
            print(f"  | {ligne}")
        tts.terminate()
        return 1
    print(f"  service TTS sur le port {port_tts}")

    port_stt, sortie_stt, stt = demarrer(["--serveur-stt", "--port", "0"], "Écoute", 300.0)
    if not port_stt:
        print(">>> ÉCHEC : le serveur STT n'a pas démarré")
        for ligne in sortie_stt.splitlines():
            print(f"  | {ligne}")
        tts.terminate()
        stt.terminate()
        return 1
    for ligne in sortie_stt.splitlines():
        if any(m in ligne for m in ("Transcription", "Écoute")):
            print(f"  | {ligne.strip()}")
    print(f"  service STT sur le port {port_stt}\n")

    base_tts, base_stt = f"http://127.0.0.1:{port_tts}", f"http://127.0.0.1:{port_stt}"
    controles: list[tuple[str, bool]] = []

    try:
        # --- 1. l'état ---------------------------------------------------------
        with urllib.request.urlopen(f"{base_stt}/sante", timeout=30) as reponse:
            sante = json.loads(reponse.read())
        print(f"  GET /sante  → {sante}")
        controles.append(("le service annonce le modèle et son état",
                          sante.get("pret") and bool(sante.get("modele"))))

        # --- 2. la parole, produite par le service TTS -------------------------
        code, wav = poster(
            f"{base_tts}/parle",
            json.dumps({"texte": PHRASE}).encode(),
            "application/json",
        )
        chemin_wav = dossier / "phrase.wav"
        chemin_wav.write_bytes(wav)
        import soundfile as sf

        echantillons, taux = sf.read(str(chemin_wav))
        print(f"\n  Kokoro a produit {len(wav)} octets de WAV "
              f"({len(echantillons) / taux:.2f} s à {taux} Hz)")
        controles.append(("le service TTS a produit l'audio", code == 200 and wav[:4] == b"RIFF"))

        # --- 3. le WAV à Whisper, par le réseau --------------------------------
        code, infos, texte = transcrire(base_stt, wav)
        trouves, total = proportions(PHRASE, texte)
        print(f"\n  POST /transcris (WAV 24 kHz mono) → HTTP {code}")
        print(f"  attendu : « {PHRASE} »")
        print(f"  entendu : « {texte} »")
        print(f"  mots retrouvés : {trouves}/{total}")
        controles.append(("la phrase dite est reconnue", trouves >= max(1, total * 0.7)))
        controles.append(("le service annonce la langue et le RTF",
                          infos.get("langue") == "fr" and infos.get("rtf", 0) >= 0))

        # --- 4. le format d'un micro de navigateur (webm/opus) -----------------
        chemin_webm = dossier / "phrase.webm"
        encodage = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(chemin_wav),
             "-c:a", "libopus", "-b:a", "32k", str(chemin_webm)],
            capture_output=True, text=True,
        )
        if encodage.returncode == 0 and chemin_webm.exists():
            code, infos, texte_webm = transcrire(base_stt, chemin_webm.read_bytes())
            trouves_webm, _ = proportions(PHRASE, texte_webm)
            print(f"\n  POST /transcris (webm/opus, {chemin_webm.stat().st_size} octets) → HTTP {code}")
            print(f"  entendu : « {texte_webm} »")
            print(f"  mots retrouvés : {trouves_webm}/{total}")
            controles.append(("le webm/opus d'un navigateur est accepté", code == 200))
            controles.append(("le webm/opus est bien transcrit", trouves_webm >= max(1, total * 0.6)))
        else:
            print(f"\n  !! ffmpeg n'a pas su encoder le webm : {encodage.stderr[:200]}")
            controles.append(("le webm/opus d'un navigateur est accepté", False))

        # --- 5. stéréo 44,1 kHz, comme un micro de machine ---------------------
        chemin_stereo = dossier / "stereo.wav"
        stereo = sf.read(str(chemin_wav), dtype="float32")[0]
        # Rééchantillonnage **réel** 24 kHz → 44,1 kHz. Écrire les mêmes échantillons en
        # déclarant 44100 Hz ne ferait qu'accélérer le son de 1,8× : le test validerait le
        # transport, mais ne dirait rien de ce qu'un vrai micro envoie.
        import numpy as np

        n = int(round(stereo.size * 44100 / 24000))
        remonte = np.interp(
            np.linspace(0, stereo.size - 1, n), np.arange(stereo.size), stereo
        ).astype(np.float32)
        sf.write(str(chemin_stereo), np_stack(remonte), 44100, subtype="PCM_16", format="WAV")
        code, _, texte_stereo = transcrire(base_stt, chemin_stereo.read_bytes())
        trouves_stereo, _ = proportions(PHRASE, texte_stereo)
        print(f"\n  POST /transcris (WAV 44,1 kHz stéréo) → HTTP {code}")
        print(f"  entendu : « {texte_stereo} »")
        print(f"  mots retrouvés : {trouves_stereo}/{total}")
        controles.append((
            "le stéréo 44,1 kHz est converti et transcrit",
            code == 200 and trouves_stereo >= max(1, total * 0.7),
        ))

        # --- 6. un contenu qui n'est pas de l'audio ---------------------------
        code, infos, _ = transcrire(base_stt, b"ceci n'est pas un son")
        print(f"\n  POST /transcris (texte au lieu de son) → HTTP {code} : {infos.get('erreur', '')[:60]}")
        controles.append(("un contenu illisible donne un 400 lisible", code == 400))

        # --- 7. le client du CLI, sur le vrai service -------------------------
        resultat = subprocess.run(
            [PY, "-m", "voicechat", "--transcrire", str(chemin_wav), "--stt-distant", base_stt],
            cwd=PROJ, capture_output=True, text=True, timeout=300, env=dict(os.environ),
        )
        sortie = resultat.stdout
        print(f"\n  voicechat --transcrire … --stt-distant → code {resultat.returncode}")
        for ligne in sortie.splitlines():
            if ligne.strip() and ("Micro" in ligne or "Résumé" in ligne or "audio" in ligne):
                print(f"  | {ligne.strip()}")
        derniere = [l for l in sortie.splitlines() if l.strip() and not l.startswith(("Micro", "Fichier", "Résumé"))]
        if derniere:
            print(f"  | texte : {derniere[-1].strip()[:100]}")
        controles.append(("le CLI transcrit via le service distant", resultat.returncode == 0))
        controles.append(("le CLI a bien transcrit la phrase",
                          proportions(PHRASE, " ".join(derniere)) [0] >= max(1, total * 0.7)))

        print("\n=== contrôles ===")
        echecs = 0
        for nom, reussi in controles:
            print(f"  {'OK ' if reussi else 'KO '} {nom}")
            echecs += 0 if reussi else 1
        print()
        if echecs:
            print(f">>> {echecs} contrôle(s) en échec")
            return 1
        print(">>> OK : texte → Kokoro → réseau → Whisper → texte, la boucle est bouclée")
        return 0
    finally:
        tts.terminate()
        stt.terminate()
        for processus in (tts, stt):
            try:
                processus.wait(timeout=20)
            except subprocess.TimeoutExpired:
                processus.kill()


def np_stack(signal):
    """Stéréo : deux canaux identiques, comme un micro qui double la même prise."""
    import numpy as np

    return np.stack([signal, signal], axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
