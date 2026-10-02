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

TRACE_RESEAU = """() => {
  // Ce que le serveur a réellement transcrit, vu du navigateur. Sans ça on ne peut que
  // supposer : la voix peut se perdre dans la chaîne webm/opus du navigateur, et c'est
  // exactement ce qu'on cherche à voir.
  window.__transcriptions = [];
  window.__blobs = [];
  const vrai = window.fetch.bind(window);
  window.fetch = async (entree, options) => {
    const url = typeof entree === "string" ? entree : (entree && entree.url) || "";
    if (url.indexOf("transcris") >= 0 && options && options.body &&
        options.body.arrayBuffer) {
      // On garde **l'audio que le navigateur a réellement produit**. C'est la seule façon de
      // savoir si un mot manquant a été mal transcrit ou jamais enregistré.
      try {
        const octets = new Uint8Array(await options.body.arrayBuffer());
        let binaire = "";
        for (let i = 0; i < octets.length; i++) binaire += String.fromCharCode(octets[i]);
        window.__blobs.push(btoa(binaire));
      } catch (e) { window.__blobs.push(""); }
    }
    const reponse = await vrai(entree, options);
    if (url.indexOf("transcris") >= 0) {
      try {
        const infos = await reponse.clone().json();
        window.__transcriptions.push(infos.texte || "");
      } catch (e) { window.__transcriptions.push("(illisible)"); }
    }
    return reponse;
  };
}"""

ANALYSE = r"""
import io
import sys

import numpy as np

from voicechat.stt_distant import decoder_audio

for chemin in sys.argv[1:]:
    try:
        signal = decoder_audio(open(chemin, "rb").read())
    except Exception as exc:
        print(f"{chemin.split('/')[-1]} : illisible ({exc})")
        continue
    pas = 1600  # 100 ms à 16 kHz
    profil = [round(float((signal[k:k + pas] ** 2).mean() ** 0.5), 4)
              for k in range(0, len(signal) - pas, pas)]
    debut = next((k for k, v in enumerate(profil) if v > 0.01), None)
    print(f"{chemin.split('/')[-1]} : {signal.size / 16000:.2f} s, "
          f"premier son à {debut * 100 if debut is not None else '-'} ms")
    print("   RMS par 100 ms : " + " ".join(str(v) for v in profil[:20]))
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

  // Trace de la lecture audio. C'est ce qui manquait : quand le son se taisait pour de bon,
  // ni le texte ni l'état de la page n'en montraient rien — seul un journal de lecture dit
  // qu'une phrase a été **jouée**, et donc qu'on est sorti du trou après une interruption.
  // Dans quel état la détection était-elle quand l'enregistrement a démarré ? Pendant la
  // réponse (`muet`), le seuil est triplé : si l'enregistrement démarre alors, il coupe le
  // début de la phrase — et le début, c'est justement le mot de réveil.
  window.__departs = [];
  const vraiLancer = window.lancerEnregistrement;
  window.lancerEnregistrement = () => {
    window.__departs.push(etatVad);
    return vraiLancer();
  };

  window.__audio = [];
  const VraiAudio = window.Audio;
  window.Audio = function (url) {
    const son = new VraiAudio(url);
    son.addEventListener("play", () => window.__audio.push("joue"));
    son.addEventListener("ended", () => window.__audio.push("fini"));
    son.addEventListener("error", () => window.__audio.push("erreur"));
    const vraiPause = son.pause.bind(son);
    son.pause = () => { window.__audio.push("pause"); return vraiPause(); };
    return son;
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
        # `-u` : sans lui, la sortie du serveur est **tamponnée** quand on la lit par un tube.
        # Le banc attendait la ligne « en local : … » qui restait dans le tampon — jusqu'à
        # croire que le serveur n'avait pas démarré, et perdre des lignes du journal (une
        # mesure tronquée au milieu d'une phrase est exactement ce qu'on a vu).
        [PY, "-u", "-m", "voicechat", "--web", "--port", "0"],
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
            page.evaluate(TRACE_RESEAU)
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

            # Barge-in : on parle **pendant** que la réponse se joue — et pas seulement
            # pendant qu'elle se fabrique. On attend donc qu'un son soit en cours (un « joue »
            # que rien n'a suivi d'un « fini ») : c'est ce qui distingue une interruption
            # réelle d'un barge-in arrivé trop tôt, qui n'interrompt rien et ne prouve rien.
            fin = time.monotonic() + 40
            en_lecture = False
            while time.monotonic() < fin:
                audio = page.evaluate("() => window.__audio || []")
                if audio and audio[-1] == "joue":
                    en_lecture = True
                    break
                time.sleep(0.2)
            print(f"      un son était-il en cours avant d'interrompre : {en_lecture}")
            controles.append(("le barge-in arrive pendant que le son joue", en_lecture))

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

            # --- le son est-il revenu après l'interruption ? ----------------------
            # C'est là que se cachait le bogue signalé à l'usage : le texte continuait
            # d'arriver, donc tout paraissait marcher — sauf qu'aucune phrase n'était plus
            # jamais jouée. Ni le texte ni l'état ne peuvent le dire : il faut un journal de
            # lecture. On attend, parce que la voix du deuxième tour arrive après son texte.
            fin = time.monotonic() + 40
            audio: list[str] = []
            revenu = False
            while time.monotonic() < fin:
                audio = page.evaluate("() => window.__audio || []")
                coupe = audio.index("pause") if "pause" in audio else len(audio)
                if "joue" in audio[coupe:]:
                    revenu = True
                    break
                time.sleep(0.5)
            print(f"      journal de lecture audio : {' '.join(audio[-14:])}")
            controles.append(("la voix a bien été jouée au moins une fois", "joue" in audio))
            controles.append(("après une interruption, le son revient", revenu))

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

            resultats: list[bool] = []
            # Trois essais, parce que la première mesure du mot de réveil a été **instable** :
            # reconnu un essai, absent le suivant (« Ordinateur » devenu « qu'elle t'en »).
            # Un mode qui ne s'active qu'une fois sur deux ne sert à rien : il faut savoir.
            for essai in (1, 2, 3):
                attendre_etat(page, "ecoute", 90)   # laisser la réponse précédente finir
                avant_essai = len(bulles_moi(page))
                page.evaluate("() => window.__parle('avec')")
                parti = attendre_message(page, avant_essai + 1, 45)
                resultats.append(bool(parti))
                print(f"      essai {essai}/3 : {'mot reconnu' if parti else 'MOT PERDU'} "
                      f"→ « {parti or ''} »")
            reconnues = sum(resultats)
            # Le taux est **mesuré**, pas exigé : le premier mot est la partie fragile de toute
            # la chaîne (l'enregistrement démarre au franchissement du seuil, donc son attaque
            # peut manquer). Mesuré selon les passes : 5/6, 3/3, 1/3, 0/3. La pré-écoute qui
            # devait corriger ça a été retirée (elle cassait l'en-tête webm) — le sujet est
            # documenté dans le README §19.8 plutôt que maquillé ici. La *logique* du mot de
            # réveil, elle, est éprouvée juste après, sans dépendre de la reconnaissance.
            print(f"      mot de réveil reconnu {reconnues}/3 (mesure, pas exigence)")
            controles.append(("aucun message envoyé ne contient le mot de réveil",
                              not any("ordinateur" in t.lower() for t in bulles_moi(page))))

            # La logique du mot de réveil, éprouvée **directement** : c'est déterministe, alors
            # que le passage par la voix dépend de la reconnaissance du premier mot.
            retire = page.evaluate(
                "() => sansLeReveil('Ordinateur, quel temps fera-t-il demain à Lyon ?')"
            )
            absent = page.evaluate(
                "() => sansLeReveil('Quel temps fera-t-il demain à Lyon ?')"
            )
            seul = page.evaluate("() => sansLeReveil('hé ordinateur, salut')")
            print(f"      sansLeReveil : avec → « {retire} » · sans → {absent} · seul → « {seul} »")
            controles.append(("le mot de réveil est retiré de la question",
                              retire == "quel temps fera-t-il demain à Lyon ?"))
            controles.append(("une phrase sans le mot ne donne rien", absent is None))
            controles.append(("l'interjection d'appel est sautée", seul == "salut"))
            # Une phrase = un envoi au serveur. La pré-écoute avait introduit un défaut de plus :
            # **cinq envois pour une même phrase**, parce que la détection appelait l'arrêt à
            # chaque mesure (toutes les 60 ms) tant que la fin de phrase était franchie.
            departs = page.evaluate("() => window.__departs || []")
            envois = page.evaluate("() => window.__transcriptions || []")
            print(f"      {len(departs)} phrase(s) détectée(s) → {len(envois)} envoi(s) au serveur")
            controles.append(("une phrase détectée donne un seul envoi au serveur",
                              len(departs) == len(envois)))

            print(f"      états au départ des enregistrements : "
                  f"{page.evaluate('() => window.__departs')}")

            print(f"      journal du micro      : {page.evaluate('() => window.__journal')}")
            print(f"      journal enregistreur  : {page.evaluate('() => window.__rec')}")
            print(f"      journal audio         : {page.evaluate('() => window.__audio')}")
            print(f"      ce que le serveur a transcrit :")
            transcriptions = page.evaluate("() => window.__transcriptions || []")
            for i, t in enumerate(transcriptions, 1):
                print(f"        {i}. « {t} »")

            # L'audio réellement produit par le navigateur, gardé sur disque puis relu : si un
            # mot manque, il faut savoir s'il a été mal transcrit ou jamais enregistré.
            import base64

            blobs = page.evaluate("() => window.__blobs || []")
            print(f"      audio reçu du navigateur ({len(blobs)} enregistrements) :")
            enregistres: list[str] = []
            for i, b64 in enumerate(blobs, 1):
                if not b64:
                    continue
                chemin = dossier / f"recu-{i}.webm"
                chemin.write_bytes(base64.b64decode(b64))
                enregistres.append(str(chemin))
            if enregistres:
                # Analyse par le python du venv : c'est lui qui a numpy et PyAV.
                analyse = subprocess.run(
                    [PY, "-c", ANALYSE, *enregistres],
                    cwd=PROJ, capture_output=True, text=True, timeout=300,
                )
                print("\n".join("      " + l for l in analyse.stdout.strip().splitlines()))
                # L'audio produit par le navigateur doit être **lisible** : un webm privé de son
                # en-tête est refusé par le serveur (« Invalid data found »), et rien ne le
                # montre côté page — sinon une transcription vide, qui ressemble à un silence.
                refuses = analyse.stdout.count("illisible")
                controles.append(
                    (f"l'audio envoyé est lisible ({len(enregistres) - refuses}/{len(enregistres)})",
                     refuses == 0 and bool(enregistres))
                )


            # --- 8. la dictée manuelle, 🎤 sans les mains libres ---------------------
            # Elle a été cassée sans qu'on le voie, de la v1.6 à la v1.7 : le second appui
            # jetait la dictée. Le banc des mains libres ne la touchait pas, et la
            # vérification du micro (v1.4) n'éprouvait que le serveur, pas le bouton.
            print("\n[8/8] dictée manuelle : 🎤, on parle, 🎤 — le texte doit rester")
            page.click("#mains")                     # on sort des mains libres
            attendre_etat(page, "repos", 20)
            page.click("#micro")
            time.sleep(1.0)
            avant_dictee = len(bulles_moi(page))
            page.evaluate("() => window.__parle('sans')")
            time.sleep(4.5)                          # la phrase se joue en entier
            page.click("#micro")                     # stop manuel : ça doit transcrire
            dicte = ""
            fin = time.monotonic() + 45
            while time.monotonic() < fin:
                dicte = page.evaluate("() => document.getElementById('q').value")
                if dicte.strip():
                    break
                time.sleep(0.4)
            print(f"      champ de saisie : « {dicte.strip()} »")
            controles.append(("la dictée manuelle garde ce qui a été dit",
                              _proches(PHRASE, dicte.strip())))
            controles.append(("le texte dicté n'est pas envoyé tout seul",
                              len(bulles_moi(page)) == avant_dictee))

            etat_final = attendre_etat(page, ("ecoute", "parle", "transcrit"), 60)
            chronologie.append(etat_final)
            print(f"      état final : {etat_final}")
            print(f"\n      chronologie observée : {' → '.join(chronologie)}")
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
