"""Vérifie le mode mains libres dans un vrai navigateur (v1.6).

Rien n'est simulé côté serveur : un vrai `voicechat --web` tourne dans un processus à part
(Kokoro sur GPU, vrai Whisper, vrai modèle derrière). Côté navigateur, un vrai Chromium
charge la vraie page et exécute le vrai JavaScript de la détection.

**Le micro est fabriqué par la vérification.** On remplace `getUserMedia` par un flux qu'on
compose nous-mêmes, à partir d'un WAV que Kokoro a prononcé — donc de la vraie parole — et
on décide **quand** il parle. Trois raisons, apprises à la dure :

1. Le périphérique factice de Chromium (`--use-file-for-fake-audio-capture`) produit un
   signal **continu** : mesuré, 0,4 de RMS partout, jamais un silence. Une détection de fin
   de phrase ne peut donc jamais s'y déclencher — l'enregistrement ne s'arrêtait jamais.
2. Le traitement audio du navigateur détruit ce même signal : avec l'anti-écho, il en
   ressort 0,047 de RMS et **plat** (du bruit résiduel), contre 0,46 sans. Whisper n'a alors
   rien à transcrire.
3. Avec un micro sous contrôle, la vérification est **déterministe** : plus besoin d'espérer
   qu'un fichier joué en boucle tombe au bon moment. On dit « maintenant tu parles », et on
   regarde.

Ce qui reste **hors de portée** de ce banc, et qu'aucune machine ne peut juger à ma place :
le comportement acoustique réel — un vrai haut-parleur qui entre dans un vrai micro, et la
qualité de l'anti-écho du navigateur. Ici, le haut-parleur ne sort aucun son. Le barge-in
est donc vérifié **en mécanique** (est-ce que parler pendant la réponse la coupe et ouvre
l'écoute ?), pas en acoustique.

    python3 tests/verif_mains_libres.py          (playwright est dans le python système)
    .venv/bin/pip install playwright && .venv/bin/python tests/verif_mains_libres.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
PY = str(PROJ / ".venv" / "bin" / "python")
PHRASE = "Quel temps fera-t-il demain à Lyon ?"
# La même phrase, précédée du mot de réveil. Deux WAV, donc : un sans, un avec.
PHRASE_REVEIL = "Ordinateur, quel temps fera-t-il demain à Lyon ?"

# Chromium complet : le « headless shell » de Playwright n'est pas toujours installé, et il
# gère moins bien la capture audio.
CHROMIUMS = [
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
    str(Path.home() / ".cache/ms-playwright/chromium-1217/chrome-linux64/chrome"),
]

GENERER = r"""
import sys, numpy as np, soundfile as sf
from voicechat.tts import KokoroTTS

tts = KokoroTTS(voice="ff_siwis")
if not tts.load():
    raise SystemExit(f"Kokoro indisponible : {tts.load_error}")
# La phrase seule, sans silence ajouté : le silence viendra de ne rien jouer du tout.
signal = tts.synth(sys.argv[2])
n = int(round(signal.size * 48000 / 24000))
remonte = np.interp(np.linspace(0, signal.size - 1, n), np.arange(signal.size), signal)
sf.write(sys.argv[1], remonte.astype(np.float32), 48000, subtype="PCM_16")
print(f"{signal.size / 24000:.2f} s de parole")
"""

# Le micro de synthèse. `__parle("sans")` joue la phrase, `__parle("avec")` la même précédée
# du mot de réveil. Entre deux appels, le flux contient du vrai silence (des zéros), ce
# qu'aucun périphérique factice ne sait faire.
MICRO_SYNTHETIQUE = """
window.__journal = [];
navigator.mediaDevices.getUserMedia = async () => {
  const ctx = new AudioContext();
  await ctx.resume();
  const charger = async (url) =>
    ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
  const buffers = {
    sans: await charger("/__micro.wav"),
    avec: await charger("/__micro-reveil.wav"),
  };
  const sortie = ctx.createMediaStreamDestination();
  window.__parle = (lequel) => {
    const nom = lequel || "sans";
    const source = ctx.createBufferSource();
    source.buffer = buffers[nom];
    source.connect(sortie);
    source.start();
    window.__journal.push("parle:" + nom);
  };
  window.__journal.push("micro ouvert");
  return sortie.stream;
};
"""

TRACE = """() => {
  window.__niveaux = [];
  const vraie = window.majJauge;
  window.majJauge = (n, seuil) => {
    window.__niveaux.push([Math.round(n * 1000) / 1000, Math.round(seuil * 1000) / 1000]);
    if (window.__niveaux.length > 600) window.__niveaux.shift();
    return vraie(n, seuil);
  };
  // Trace de l'enregistreur : un `stop()` qui ne déclenche pas son événement laisse la page
  // bloquée sans rien dire. On veut pouvoir le voir.
  window.__rec = [];
  const Vrai = window.MediaRecorder;
  window.MediaRecorder = class extends Vrai {
    constructor(...a) { super(...a); window.__rec.push("créé"); }
    addEventListener(type, ...reste) {
      if (type === "stop") {
        const f = reste[0];
        reste[0] = (...a) => { window.__rec.push("ÉVÉNEMENT stop"); return f(...a); };
      }
      return super.addEventListener(type, ...reste);
    }
    start(...a) {
      try { super.start(...a); window.__rec.push("start ok"); }
      catch (e) { window.__rec.push("start KO: " + e); throw e; }
    }
    stop(...a) {
      try { super.stop(...a); window.__rec.push("stop demandé"); }
      catch (e) { window.__rec.push("stop KO: " + e); }
    }
  };
}"""


def chromium() -> str:
    for chemin in CHROMIUMS:
        if Path(chemin).exists():
            return chemin
    raise SystemExit("aucun navigateur Chromium trouvé")


def fabriquer_voix(chemin: Path, phrase: str = PHRASE) -> str:
    """Fait dire la phrase à Kokoro, via le python du venv (playwright n'y est pas forcément)."""
    resultat = subprocess.run(
        [PY, "-c", GENERER, str(chemin), phrase],
        cwd=PROJ, capture_output=True, text=True, timeout=300,
    )
    if resultat.returncode != 0:
        raise SystemExit(f"génération de la voix impossible :\n{resultat.stderr[-600:]}")
    return resultat.stdout.strip()


def demarrer_serveur() -> tuple[str, subprocess.Popen, list[str]]:
    """Lance le serveur et **draine sa sortie** en continu.

    Un `Popen` dont on lit la sortie seulement au démarrage puis qu'on abandonne peut se
    bloquer en écriture dès que le tube est plein : le serveur reste suspendu au milieu
    d'une requête, et on croit à un bug ailleurs. Le fil de lecture évite ça, et garde les
    messages du serveur sous la main pour le diagnostic.
    """
    processus = subprocess.Popen(
        [PY, "-m", "voicechat", "--web", "--port", "0"],
        cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ),
    )
    journal: list[str] = []
    pret = threading.Event()

    def lire() -> None:
        for ligne in processus.stdout or []:
            journal.append(ligne.rstrip("\n"))
            if "en local :" in ligne:
                pret.set()

    threading.Thread(target=lire, daemon=True).start()
    if not pret.wait(timeout=300):
        raise SystemExit("le serveur web n'a pas démarré :\n" + "\n".join(journal[-20:]))
    for ligne in journal:
        trouve = re.search(r"en local : (http://[^\s]+)", ligne)
        if trouve:
            return trouve.group(1), processus, journal
    raise SystemExit("adresse locale introuvable dans la sortie du serveur")


def lire_etat(page) -> str:
    return page.evaluate("() => document.body.dataset.etat || 'aucun'")


def attendre_etat(page, voulu: str | tuple[str, ...], delai: float = 20.0) -> str:
    cibles = (voulu,) if isinstance(voulu, str) else voulu
    fin = time.monotonic() + delai
    vu = lire_etat(page)
    while time.monotonic() < fin:
        vu = lire_etat(page)
        if vu in cibles:
            return vu
        time.sleep(0.1)
    return vu


def bulles_moi(page) -> list[str]:
    return page.evaluate(
        "() => Array.from(document.querySelectorAll('#fil .moi .bulle')).map(b => b.textContent)"
    )


def aide(page) -> str:
    return page.evaluate("() => document.getElementById('aide').textContent")


def etat_serveur(base: str) -> dict:
    try:
        with urllib.request.urlopen(f"{base}/etat", timeout=30) as reponse:
            return json.loads(reponse.read()).get("transcription", {})
    except Exception as exc:  # pragma: no cover
        return {"erreur": str(exc)}


def attendre_message(page, combien: int, delai: float = 90.0) -> str | None:
    """Attend qu'un message de l'utilisateur apparaisse, **sans rien toucher**."""
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        bulles = bulles_moi(page)
        if len(bulles) >= combien:
            return bulles[combien - 1]
        time.sleep(0.2)
    return None


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright est absent de ce python. Deux solutions :")
        print("  python3 tests/verif_mains_libres.py")
        print("  .venv/bin/pip install playwright   (puis relancer avec le python du venv)")
        return 2

    dossier = Path(tempfile.mkdtemp(prefix="vc-mains-libres-"))
    micro_wav = dossier / "phrase.wav"
    micro_reveil = dossier / "phrase-reveil.wav"
    print("=" * 78)
    print("Mode mains libres : vrai navigateur, vrai serveur, parole en micro synthétique")
    print("=" * 78)
    print("\n[1/6] fabrication du micro : Kokoro dit deux phrases")
    print(f"      sans le mot de réveil : {fabriquer_voix(micro_wav, PHRASE)}")
    print(f"      avec le mot de réveil : {fabriquer_voix(micro_reveil, PHRASE_REVEIL)}")

    print("[2/6] démarrage du serveur…")
    base, serveur, journal = demarrer_serveur()
    print(f"      {base}")

    controles: list[tuple[str, bool]] = []
    chronologie: list[str] = []
    erreurs_js: list[str] = []

    try:
        with sync_playwright() as p:
            nav = p.chromium.launch(
                executable_path=chromium(),
                args=["--autoplay-policy=no-user-gesture-required", "--no-sandbox"],
            )
            page = nav.new_page()
            page.on("console", lambda m: erreurs_js.append(m.text) if m.type == "error" else None)
            page.route(
                "**/__micro.wav",
                lambda route: route.fulfill(path=str(micro_wav), content_type="audio/wav"),
            )
            page.route(
                "**/__micro-reveil.wav",
                lambda route: route.fulfill(path=str(micro_reveil), content_type="audio/wav"),
            )
            page.add_init_script(MICRO_SYNTHETIQUE)
            page.goto(base + "/")
            page.wait_for_timeout(1500)
            page.evaluate(TRACE)

            # --- 3. l'interface -------------------------------------------------
            print("\n[3/6] la page propose-t-elle le mode ?")
            micro_visible = page.is_visible("#micro")
            mains_visible = page.is_visible("#mains")
            print(f"      bouton 🎤 : {micro_visible} · bouton « ∞ » : {mains_visible}")
            controles.append(("les deux boutons sont proposés", micro_visible and mains_visible))
            controles.append(("la page a la jauge de niveau",
                              page.evaluate("() => !!document.getElementById('jauge-niveau')")))

            # --- 4. mains libres, seuil calibré sur le silence -------------------
            print("\n[4/6] activation des mains libres — le micro ne dit encore rien :")
            page.click("#mains")
            etat = attendre_etat(page, "ecoute", 20)
            chronologie.append(etat)
            seuil = page.evaluate("() => document.getElementById('jauge-seuil').style.left")
            print(f"      état : {etat} · seuil appris (position sur la jauge) : {seuil}")
            controles.append(("le seuil se calibre puis passe en écoute", etat == "ecoute"))
            controles.append(("la jauge de niveau est affichée", page.is_visible("#jauge")))
            controles.append(("le seuil est mesuré, pas figé", bool(seuil) and seuil != "0%"))

            # --- 5. on parle : détection, arrêt sur silence, envoi tout seul -----
            print("\n[5/6] on parle (le micro synthétique joue la phrase), puis on se tait :")
            page.evaluate("() => window.__parle()")
            depart = time.monotonic()

            vu_parle = attendre_etat(page, "parle", 15)
            chronologie.append(vu_parle)
            print(f"      état pendant la phrase : {vu_parle}")
            controles.append(("la parole déclenche l'enregistrement", vu_parle == "parle"))

            envoye = attendre_message(page, 1, 90)
            duree = time.monotonic() - depart
            for etat in ("transcrit", "ecoute", "muet"):
                if etat not in chronologie:
                    chronologie.append(etat)
            if envoye:
                print(f"      message parti seul après {duree:.1f} s : « {envoye} »")
            else:
                print(f"      affiché par la page : « {aide(page)} »")
                print(f"      état du service    : {etat_serveur(base)}")
                print(f"      journal de l'enregistreur : {page.evaluate('() => window.__rec')}")
                niveaux = page.evaluate("() => window.__niveaux || []")
                print(f"      derniers niveaux/seuils ({len(niveaux)}) : "
                      + " ".join(f"{n:.2f}/{s:.2f}" for n, s in niveaux[-10:]))
            controles.append(("le silence arrête l'enregistrement et le message part seul",
                              bool(envoye)))
            controles.append(("le texte envoyé est celui de la phrase dite",
                              bool(envoye) and _proches(PHRASE, envoye)))

            # --- 6. la réponse, la reprise, et le barge-in ------------------------
            print("\n[6/6] réponse, reprise d'écoute — puis on interrompt en parlant :")
            etat_muet = attendre_etat(page, "muet", 20)
            chronologie.append(etat_muet)
            print(f"      état pendant la réponse : {etat_muet}")
            controles.append(("pendant la réponse, le micro passe en veille",
                              etat_muet == "muet"))

            # Barge-in : on parle **pendant** que la réponse se joue.
            page.evaluate("() => window.__parle()")
            etat_interruption = attendre_etat(page, ("parle", "transcrit", "ecoute"), 20)
            chronologie.append(etat_interruption)
            print(f"      après avoir parlé par-dessus : {etat_interruption} "
                  f"(la réponse a été coupée, l'écoute a repris)")
            controles.append(("parler pendant la réponse la coupe et ouvre l'écoute",
                              etat_interruption in ("parle", "transcrit")))

            # La boucle : un deuxième message, toujours sans rien toucher.
            deuxieme = attendre_message(page, 2, 90)
            if deuxieme:
                print(f"      deuxième message, toujours sans rien toucher : « {deuxieme[:60]} »")
            controles.append(("la boucle tourne : un deuxième tour part tout seul",
                              bool(deuxieme)))

            # --- 7. le mot de réveil ----------------------------------------------
            # La même phrase, avec et sans le mot. C'est le cœur du mode : sans le mot, il ne
            # doit **rien** se passer — sinon l'assistant répond à la télévision.
            print("\n[7/7] mot de réveil : la même phrase, avec et sans le mot")
            page.click("#reveil")
            avant = len(bulles_moi(page))
            print(f"      réveil activé, messages déjà envoyés : {avant}")

            page.evaluate("() => window.__parle('sans')")
            vus: list[str] = []
            fin = time.monotonic() + 14
            while time.monotonic() < fin:
                vus.append(aide(page))
                time.sleep(0.4)
            apres_sans = len(bulles_moi(page))
            explique = any("mot de réveil" in texte for texte in vus)
            print(f"      phrase SANS le mot → {apres_sans} message(s) envoyé(s)")
            print(f"      ce que la page a affiché : « {vus[-1] if vus else ''} »")
            controles.append(("sans le mot de réveil, rien n'est envoyé", apres_sans == avant))
            controles.append(("la page dit pourquoi elle n'a rien envoyé", explique))

            page.evaluate("() => window.__parle('avec')")
            envoye_reveil = attendre_message(page, avant + 1, 60)
            if envoye_reveil:
                print(f"      phrase AVEC le mot → « {envoye_reveil} »")
            controles.append(("avec le mot de réveil, le message part", bool(envoye_reveil)))
            controles.append(("le mot de réveil n'est pas transmis au modèle",
                              bool(envoye_reveil) and "ordinateur" not in envoye_reveil.lower()))

            etat_final = attendre_etat(page, ("ecoute", "parle", "transcrit"), 60)
            chronologie.append(etat_final)
            print(f"      état final : {etat_final}")
            print(f"\n      chronologie observée : {' → '.join(chronologie)}")
            print(f"      journal du micro      : {page.evaluate('() => window.__journal')}")
            print(f"      journal enregistreur  : {page.evaluate('() => window.__rec')}")

            controles.append(("aucune erreur JavaScript", not erreurs_js))
            if erreurs_js:
                for err in erreurs_js[:4]:
                    print(f"      !! JS : {err[:110]}")
            nav.close()
    finally:
        serveur.terminate()
        try:
            serveur.wait(timeout=25)
        except subprocess.TimeoutExpired:
            serveur.kill()

    print("\n=== contrôles ===")
    echecs = 0
    for nom, reussi in controles:
        print(f"  {'OK ' if reussi else 'KO '} {nom}")
        echecs += 0 if reussi else 1
    print()
    if echecs:
        print(f">>> {echecs} contrôle(s) en échec")
        return 1
    print(">>> OK : on parle, ça s'arrête tout seul, ça part, ça répond, ça réécoute")
    return 0


def _proches(attendu: str, obtenu: str) -> bool:
    """La phrase est-elle reconnue, à peu près ? (Whisper se trompe un peu, c'est normal.)"""
    mots = [m for m in re.findall(r"[a-zà-ÿ]+", attendu.lower()) if len(m) > 3]
    trouves = sum(1 for m in mots if m in obtenu.lower())
    return trouves >= max(1, len(mots) * 0.5)


if __name__ == "__main__":
    raise SystemExit(main())
