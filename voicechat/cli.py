"""Boucle de chat en console : orchestre LLM (streaming) → phrases → voix."""

from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
from pathlib import Path

from . import __version__, store, text as txt
from .audio import Speaker
from .cache import CacheAudio
from .config import (
    DEFAULT_LANG,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    Config,
    lire_config,
    nom_court,
)
from .distant import PORT_DEFAUT, TTSDistant
from .editor import ClavierGeneration, LineEditor
from .llm import (
    Connexion,
    LLMError,
    contexte_du_serveur,
    list_models,
    premier_serveur,
    stream_with_usage,
)
from .micro import DetecteurParole, Ecouteur, Microphone
from .stt import Transcriber, Transcription
from .stt_distant import PORT_STT, STTDistant
from .tour import derouler
from .tts import (
    KokoroTTS,
    SpeechPipeline,
    Synthetiseur,
    analyser_voix,
    decrire_voix,
    est_melange,
    list_voices,
    resolve_device,
)
from .web import PORT_WEB

BANNER = r"""
 __     __    _         _____ _           _
 \ \   / /__ (_) ___ ___|_   _| |__   __| | __ _
  \ \ / / _ \| |/ __/ _ \| | | '_ \ / _` |/ _` |
   \ V / (_) | | (_|  __/| | | | | | (_| | (_| |
    \_/ \___/|_|\___\___||_| |_| |_|\__,_|\__,_|
"""

# Consigne donnée au modèle quand on lui demande de compacter l'historique (/resume).
# Elle insiste sur « n'invente rien » : un résumé qui brode ferait entrer dans le
# contexte des faits que l'utilisateur n'a jamais dits, et le modèle les tiendrait
# ensuite pour vrais.
PROMPT_RESUME = (
    "Tu résumes une conversation pour en garder la mémoire. Écris en français, à la "
    "troisième personne, en paragraphes continus — sans titres, sans listes à puces, "
    "sans gras : ces artifices consomment des tokens sans rien apprendre. Conserve : "
    "les sujets abordés, les décisions prises, les préférences exprimées, les noms "
    "propres et les faits techniques (chiffres, chemins, versions). Sois dense et "
    "factuel. N'invente rien — ce qui n'a pas été dit ne doit pas apparaître."
)

def _jetons_estimes(texte: str) -> int:
    """Estimation du nombre de tokens d'un texte, en français.

    ~3,5 caractères par token est l'ordre de grandeur usuel pour du français avec un
    tokenizer BPE. Ce n'est qu'un **repli** : dès qu'un tour a eu lieu, le client prend
    le compte exact annoncé par le serveur et n'estime que ce qui a été ajouté depuis.
    """
    return int(len(texte) / 3.5) + 1


# Source unique de vérité : sert à la fois à /help et à la complétion de Tab.
COMMANDES: list[tuple[str, str]] = [
    ("/help", "cette aide"),
    ("/quit", "quitter (alias /exit, /q)"),
    ("/reset", "vide l'historique de conversation"),
    ("/resume", "compacte les vieux échanges en un résumé (au lieu de les perdre)"),
    ("/contexte", "où en est le contexte : tokens utilisés, place restante"),
    ("/voice <nom>", "voix Kokoro, ou mélange : « ff_siwis:3+ef_dora:1 » (ex. /voice af_heart)"),
    ("/lang <code>", "langue Kokoro : a en, b en-GB, f fr, e es, i it, p pt, j ja, z zh, h hi"),
    ("/speed <x>", "vitesse de lecture (ex. /speed 1.15)"),
    ("/tts on|off", "active ou coupe la voix"),
    ("/model <nom>", "change de modèle pour les tours suivants"),
    ("/system <texte>", "remplace le prompt système"),
    ("/profil [nom]", "liste les profils de prompt système, ou en charge un"),
    ("/profil save <nom>", "enregistre le prompt système courant comme profil"),
    ("/save [nom]", "enregistre la conversation (défaut : derniere)"),
    ("/load <nom>", "recharge une conversation sauvegardée"),
    ("/conversations", "liste les conversations sauvegardées"),
    ("/cherche <mot>", "cherche un passage dans les conversations sauvegardées (insensible aux accents)"),
    ("/forget <nom>", "supprime une conversation sauvegardée"),
    ("/export [fichier]", "écrit la conversation en markdown"),
    ("/voices", "liste les voix Kokoro (nécessite Hugging Face)"),
    ("/device", "liste les sorties audio détectées"),
    ("/ecoute", "écouter le micro et envoyer ce qui est dit"),
    ("/micro on|off", "mode mains libres : écoute après chaque réponse"),
    ("/rejoue", "réentendre les phrases non synthétisées (GPU revenu)"),
    ("/cache", "état de la réserve d'audio déjà synthétisé (`/cache vider`)"),
    ("/stats", "dernières mesures (latence, débit, RTF du TTS)"),
    ("/debug", "bascule l'affichage des stats à chaque tour"),
]

NOMS_COMMANDES: list[str] = sorted({nom.split()[0] for nom, _ in COMMANDES})

_AIDE_CLAVIER = """
Entrée vocale :
  voicechat --micro          mode mains libres : le micro remplace le clavier
  /ecoute                    dicter un seul message, puis revenir au clavier
  /micro on|off              activer ou couper le mode mains libres
  (Ctrl+C pendant l'écoute rend la main au clavier)
  voicechat --transcrire f.wav   transcrire un fichier, sans conversation

Pendant une réponse :
  Échap              couper la voix, mais garder la réponse à l'écran
  Ctrl+C             tout couper : génération, synthèse et lecture
  (vos frappes pendant la réponse sont conservées pour le message suivant)

Clavier :
  Coller un texte multi-lignes : il est conservé tel quel, rien n'est envoyé
                                 tant que vous n'avez pas frappé Entrée
  ⏎ Entrée        envoyer le message
  ↑ / ↓            historique des messages
  Ctrl+R           rechercher dans l'historique (Ctrl+R à nouveau : plus ancien,
                   Ctrl+G pour abandonner)
  Tab              compléter les commandes et leurs arguments
  ← / →            déplacer le curseur · Ctrl+U effacer · Ctrl+W effacer le mot
  Ctrl+C           effacer le brouillon (sur ligne vide : quitter)
  Ctrl+D           quitter"""

HELP = (
    "Commandes disponibles :\n"
    + "\n".join(f"  {nom:<20} {aide}" for nom, aide in COMMANDES)
    + "\n"
    + _AIDE_CLAVIER
)



# --------------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicechat",
        description="Chat console avec un LLM local, réponses synthétisées par Kokoro.",
    )
    p.add_argument("--base-url", help="racine de l'API OpenAI-compatible (ex. http://hote:8080/v1)")
    p.add_argument("--secours", help="serveurs de secours, séparés par des virgules (essayés dans l'ordre)")
    p.add_argument("--model", help="nom du modèle (défaut : détecté via /v1/models)")
    p.add_argument("--api-key", help="jeton Bearer si le serveur en exige un")
    p.add_argument("--system", help="prompt système")
    p.add_argument("--voice", help="voix Kokoro (ff_siwis), ou mélange : ff_siwis:3+ef_dora:1")
    p.add_argument("--lang", help="code langue Kokoro (a=en, b=en-GB, f=fr, e=es, i=it, p=pt, j=ja, z=zh, h=hi)")
    p.add_argument("--speed", type=float, help="vitesse de lecture (1.0 = normal)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], help="périphérique de synthèse")
    p.add_argument("--output-device", help="index ou nom du périphérique de sortie audio")
    p.add_argument("--no-tts", action="store_true", help="désactiver complètement la voix")
    p.add_argument("--no-cache", action="store_true",
                   help="ne pas réutiliser l'audio des phrases déjà synthétisées")
    p.add_argument("--tts-distant", metavar="URL",
                   help="synthétiser sur un autre poste (ex. http://niko-tv:8090)")
    p.add_argument("--serveur-tts", action="store_true",
                   help="tenir le service TTS pour d'autres postes, puis sortir")
    p.add_argument("--serveur-stt", action="store_true",
                   help="tenir le service de transcription pour d'autres postes, puis sortir")
    p.add_argument("--stt-distant", metavar="URL",
                   help="transcrire sur un autre poste (ex. http://niko-tv:8092)")
    p.add_argument("--web", action="store_true",
                   help="servir l'interface web (page + texte au fil de l'eau + audio)")
    p.add_argument("--port", type=int, default=None,
                   help=f"port d'écoute (défaut : {PORT_DEFAUT} pour --serveur-tts, "
                        f"{PORT_STT} pour --serveur-stt, {PORT_WEB} pour --web)")
    p.add_argument("--temperature", type=float, help="température d'échantillonnage")
    p.add_argument("--timeout", type=float, help="délai max en secondes par requête")
    p.add_argument("--probe", action="store_true", help="tester la joignabilité du serveur et sortir")
    p.add_argument("--list-voices", action="store_true", help="lister les voix Kokoro et sortir")
    p.add_argument("--continue", dest="reprendre", action="store_true",
                   help="reprendre la dernière conversation sauvegardée")
    p.add_argument("--profil", help="profil de prompt système à charger au démarrage")
    p.add_argument("--micro", action="store_true",
                   help="mains libres : écoute le micro après chaque réponse")
    p.add_argument("--stt-modele", help="modèle Whisper (tiny, base, small, medium, large-v3)")
    p.add_argument("--stt-device", choices=["auto", "cuda", "cpu"], help="périphérique de transcription")
    p.add_argument("--stt-langue", help="langue parlée (fr, en…), vide = détection automatique")
    p.add_argument("--micro-device", help="index ou nom du périphérique d'entrée audio")
    p.add_argument("--transcrire", metavar="FICHIER",
                   help="transcrire un fichier audio et sortir (pas de conversation)")
    p.add_argument("--diag-micro", action="store_true",
                   help="contrôler le micro (niveau, saturation, réaction du VAD) et sortir")
    p.add_argument("--dire", metavar="TEXTE",
                   help="lire un texte à voix haute sans le LLM, puis sortir")
    p.add_argument("--dire-fichier", metavar="FICHIER",
                   help="lire un fichier texte à voix haute, puis sortir")
    p.add_argument("-q", "--question", metavar="TEXTE",
                   help="poser une seule question, répondre et sortir (mode scriptable)")
    p.add_argument("--debug", action="store_true", help="afficher les statistiques à chaque tour")
    p.add_argument("--version", action="version", version=f"voicechat {__version__}")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    # Le fichier TOML est lu ici, une seule fois : ses réglages servent de valeurs par
    # défaut, que les variables d'environnement puis les arguments surclassent.
    base, avertissement = lire_config()
    if avertissement:
        print(f"[config] {avertissement}")
    cfg = Config.from_env(base)
    out_dev = args.output_device
    if out_dev is not None:
        try:
            out_dev = int(out_dev)
        except (TypeError, ValueError):
            pass  # nom de périphérique : sounddevice accepte aussi une chaîne
    return cfg.with_overrides(
        base_url=args.base_url.rstrip("/") if args.base_url else None,
        secours=(
            [u.strip().rstrip("/") for u in args.secours.replace(";", ",").split(",") if u.strip()]
            if args.secours
            else None
        ),
        model=args.model,
        api_key=args.api_key,
        system=args.system,
        voice=args.voice,
        lang=args.lang,
        speed=args.speed,
        device=args.device,
        output_device=out_dev,
        tts=False if args.no_tts else None,
        cache=False if args.no_cache else None,
        tts_url=args.tts_distant.rstrip("/") if args.tts_distant else None,
        stt_url=args.stt_distant.rstrip("/") if args.stt_distant else None,
        temperature=args.temperature,
        timeout=args.timeout,
        profil=args.profil,
        micro=True if args.micro else None,
        stt_modele=args.stt_modele,
        stt_device=args.stt_device,
        stt_langue=args.stt_langue,
        micro_device=args.micro_device,
        show_stats=True if args.debug else None,
    )


# ----------------------------------------------------------------------- modes courts
def do_probe(cfg: Config) -> int:
    """Teste chaque serveur dans l'ordre et dit lequel répond.

    Tester les cibles une par une (au lieu de s'arrêter à la première qui échoue)
    est justement l'intérêt : on veut savoir ce qui est joignable et ce qui ne
    l'est pas, pas seulement si « ça marche ».
    """
    cibles = cfg.cibles
    print(f"Cibles    : {len(cibles)} serveur(s), essayés dans l'ordre")
    premier: str | None = None
    for cible in cibles:
        marque = "(principal)" if cible == cfg.base_url else "(secours)"
        t0 = time.monotonic()
        try:
            modeles = list_models(cible, cfg.api_key, timeout=10.0)
        except LLMError as exc:
            print(f"  ÉCHEC {cible} {marque} en {time.monotonic() - t0:.1f} s")
            for ligne in str(exc).splitlines():
                print(f"        {ligne}")
            continue
        print(f"  OK    {cible} {marque} en {time.monotonic() - t0:.2f} s "
              f"— {len(modeles)} modèle(s)")
        if premier is None:
            premier = cible
            for nom in modeles:
                print(f"        • {nom}")
    if premier is None:
        print("\nAucun serveur ne répond.")
        print("  → machine éteinte, tailscale down, ou serveur lancé sans --host 0.0.0.0")
        return 2
    if premier != cfg.base_url:
        print(f"\nLa cible principale ne répond pas : c'est « {premier} » qui servira.")
    return 0


def charger_transcripteur(cfg: Config):
    """Prépare la transcription : locale, ou déportée si ``stt_url`` est renseigné.

    Retourne le transcripteur prêt, ou ``None`` après avoir affiché pourquoi. Les deux
    sites d'appel — le mode `--transcrire` et le micro de la session — veulent exactement
    le même comportement ; les laisser écrire chacun leur tour de création garantissait
    qu'ils finissent par diverger.

    Les deux implémentations se présentent pareil (``pret``, ``charger()``,
    ``transcrire()``, ``transcrire_fichier()``) : le reste du programme n'a pas à savoir
    laquelle il tient.
    """
    if cfg.stt_url:
        print(f"Micro  : connexion au service STT {cfg.stt_url}…")
        transcriber = STTDistant(url=cfg.stt_url, langue=cfg.stt_langue)
    else:
        print(f"Micro  : chargement de whisper « {cfg.stt_modele} »…")
        transcriber = Transcriber(
            modele=cfg.stt_modele, device=cfg.stt_device, langue=cfg.stt_langue
        )

    if not transcriber.charger():
        print(f"Micro  : INDISPONIBLE — {transcriber.erreur}")
        return None
    print(f"Micro  : {decrire_transcripteur(cfg, transcriber)}")
    return transcriber


def decrire_transcripteur(cfg: Config, transcriber) -> str:
    """La ligne de console qui dit ce qui transcrit, et où."""
    if cfg.stt_url:
        return f"transcription déportée vers {cfg.stt_url} (modèle {transcriber.modele})"
    return f"whisper « {cfg.stt_modele} » sur {transcriber.device} ({transcriber.compute_reel})"


def do_transcrire(cfg: Config, chemin: str) -> int:
    """Mode one-shot : transcrire un fichier audio, sans conversation ni voix."""
    import os

    if not os.path.isfile(chemin):
        print(f"fichier introuvable : {chemin}")
        return 2

    transcriber = charger_transcripteur(cfg)
    if transcriber is None:
        return 2

    try:
        resultat = transcriber.transcrire_fichier(chemin)
    except Exception as exc:
        print(f"[ERREUR] {type(exc).__name__} : {exc}")
        return 2

    print(f"Fichier: {chemin}")
    print(f"Résumé : {resultat.resume()}")
    print()
    print(resultat.texte or "(rien transcrit)")
    return 0


def do_diag_micro(cfg: Config) -> int:
    """Contrôle de santé du micro : niveau, saturation, et réaction du VAD."""
    import numpy as np

    vad = DetecteurParole()
    if not vad.pret:
        print(f"[ERREUR] {vad.erreur}")
        return 2

    micro = Microphone(device=cfg.micro_device)
    if not micro.available:
        print(f"[ERREUR] {micro.erreur or 'entrée audio indisponible'}")
        return 2

    print("Micro  : mesure du bruit de fond pendant 3 s (ne parle pas)…")
    blocs: list = []
    probas: list[float] = []
    with micro:
        if not micro.pret:
            print(f"[ERREUR] {micro.erreur}")
            return 2
        for _ in range(int(3 * 16000 / DetecteurParole.TAILLE_BLOC)):
            bloc = micro.lire(1.0)
            if bloc is None:
                break
            blocs.append(bloc)
            probas.append(vad.probabilite(bloc))

    if not blocs:
        print("[ERREUR] aucun échantillon reçu du micro")
        return 2

    x = np.concatenate(blocs)
    rms = float(np.sqrt((x**2).mean()))
    db = 20 * np.log10(rms + 1e-12)
    sature = float((np.abs(x) > 0.99).mean()) * 100
    parole = sum(1 for p in probas if p >= vad.seuil)

    print(f"  échantillons : {x.size} ({x.size / 16000:.1f} s)")
    print(f"  niveau       : RMS {rms:.4f} ({db:.1f} dBFS)")
    print(f"  écrêtage     : {sature:.2f} % des échantillons")
    print(
        f"  VAD (seuil {vad.seuil}) : max {max(probas):.3f}, moyen {sum(probas) / len(probas):.3f}"
        f" — {parole}/{len(probas)} blocs jugés parole"
    )
    print()

    if sature > 1.0:
        print("  ⚠ micro SATURÉ : la reconnaissance sera mauvaise. Baisse le gain :")
        print("      amixer -c 0 sget Capture      # voir la valeur actuelle")
        print("      amixer -c 0 sset Capture 25   # ~40 % : bon compromis mesuré")
        print("    (gain à 100 % = 30 dB : le micro écrête, le signal devient du bruit)")
    elif db < -50:
        print("  ⚠ signal très faible : monte un peu le gain de capture.")
    else:
        print("  ✓ niveau correct")

    if parole == 0:
        print("  ℹ aucune parole pendant la mesure : c'est normal si tu t'es tu.")
        print("    Relance en parlant pour vérifier que le VAD réagit.")
    print()
    print("  dicter un message : /ecoute      (dans une session)")
    print("  mains libres      : voicechat --micro")
    return 0


def do_list_voices() -> int:
    voix = list_voices()
    if not voix:
        print("Impossible de récupérer la liste (Hugging Face injoignable ?).")
        return 1
    print(f"{len(voix)} voix Kokoro — préfixe : a/b = anglais, f = français, e = espagnol,")
    print("i = italien, p = portugais, j = japonais, z = chinois, h = hindi\n")
    par_langue: dict[str, list[str]] = {}
    for v in voix:
        par_langue.setdefault(v[0], []).append(v)
    for code in sorted(par_langue):
        print(f"  [{code}] " + ", ".join(par_langue[code]))
    print()
    print("Mélanger deux voix : /voice ff_siwis,ef_dora   (moitié-moitié)")
    print("Pondérer :           /voice ff_siwis:3+ef_dora:1   (75 % / 25 %)")
    print("Mélanger des voix d'autres langues est permis : Kokoro prévient, on le tait.")
    print(f"Écouter des exemples : .venv/bin/python tests/bench_voix.py")
    return 0


# --------------------------------------------------------------------- la session
class ChatSession:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Déclarés avant tout le reste : l'application du profil (juste en dessous)
        # passe par `_reglages_profil`, qui regarde `self.tts` pour savoir si le TTS
        # est déjà chargé. Sans ça, --profil plantait sur un AttributeError.
        self.tts: Synthetiseur | None = None
        self.speaker: Speaker | None = None
        self.speech: SpeechPipeline | None = None
        self.cache: CacheAudio | None = None
        self.profil = ""
        if cfg.profil:  # --profil : le prompt système vient d'un fichier
            try:
                profil = store.lire_profil(cfg.profil)
                if not profil.prompt:
                    raise ValueError(f"le profil « {cfg.profil} » ne contient aucun prompt")
                cfg.system = profil.prompt
                self._reglages_profil(profil)  # voix / vitesse / langue, si fournis
                self.profil = store.nom_fichier(cfg.profil)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[profil] {exc}")
        self.messages: list[dict] = [{"role": "system", "content": cfg.system}]
        self.editor = LineEditor(completeur=self._completeur)
        self.last_stats = ""
        self.serveur_actif = ""  # renseigné si on a dû basculer sur un secours
        self._attente_vue = 0  # phrases non synthétisées déjà signalées
        self.conversation = ""  # nom de la conversation courante, pour /save
        self.debut = time.time()
        self.frappes = ""  # frappes faites pendant la réponse, rendues au prochain prompt
        self.modeles: list[str] = []  # noms connus du serveur, pour Tab
        self._voix: list[str] | None = None  # voix Kokoro, récupérées une seule fois
        self.transcripteur: Transcription | None = None  # chargé au premier besoin
        self.ecouteur: Ecouteur | None = None
        self._stt_indisponible = False
        self.mains_libres = cfg.micro
        # Un seul avertissement par « vague » d'oublis : sinon l'historique plein
        # répéterait le même message à chaque tour.
        self._oubli_signale = False
        self._contexte_signale = False
        # Ancrage du comptage de tokens : le compte exact donné par le serveur au tour
        # précédent, et la taille qu'avait l'historique à cet instant. On n'estime que ce
        # qui a été ajouté depuis — deux messages, donc une erreur minuscule.
        self._ancrage_jetons = 0
        self._ancrage_messages = 0

    # ------------------------------------------------------------------ démarrage
    def setup_voice(self, avec_pipeline: bool = True) -> None:
        """Charge la voix.

        ``avec_pipeline=False`` sert au serveur web : la synthèse doit être disponible
        (c'est elle qui fournit l'audio au navigateur) mais **la machine ne doit pas
        parler** — le son sort du navigateur, pas de ses haut-parleurs.
        """
        cfg = self.cfg
        if not cfg.tts:
            print("Voix   : désactivée (--no-tts)")
            return

        if avec_pipeline:
            self.speaker = Speaker(samplerate=24000, device=cfg.output_device)  # type: ignore[arg-type]
            if not self.speaker.available:
                print("Voix   : sounddevice indisponible → chat muet")
                self.cfg.tts = False
                return
            self.speaker.start()
            if not self.speaker.pret:
                print(f"Voix   : {self.speaker.last_error} → chat muet")
                self.cfg.tts = False
                return
        else:
            print("Voix   : synthèse seule — le son sortira du navigateur")

        if cfg.tts_url:
            # Poste léger : aucune synthèse locale, donc ni torch ni GPU ici.
            print(f"Voix   : synthèse déportée vers {cfg.tts_url}")
            self.tts = TTSDistant(
                url=cfg.tts_url,
                voice=cfg.voice,
                lang_code=cfg.lang,
                speed=cfg.speed,
                timeout=max(30.0, cfg.timeout),
            )
            if not self.tts.load():
                print(f"Voix   : {self.tts.load_error}")
                print("         (le chat continue en texte)")
                self.cfg.tts = False
                self.tts = None
                return
            print(f"Distant: {self.tts.device_reason}")
            self._brancher_pipeline()
            return

        print(f"Voix   : chargement de Kokoro « {decrire_voix(cfg.voice)} »…")
        self.cache = CacheAudio(actif=cfg.cache)
        self.tts = KokoroTTS(
            voice=cfg.voice,
            lang_code=cfg.lang,
            speed=cfg.speed,
            device=cfg.device,
            cache=self.cache,
        )
        if not self.tts.load():
            print(f"Voix   : INDISPONIBLE — {self.tts.load_error}")
            print("         (le chat continue en texte ; voir README §8)")
            self.cfg.tts = False
            self.tts = None
            return

        print(f"GPU    : {self.tts.device_reason} → device={self.tts.device}")
        if self.cache is not None:
            etat = "active" if cfg.cache else "désactivée (--no-cache)"
            print(f"Réserve: {etat} — {self.cache.dossier}")
        self._brancher_pipeline()

    def _brancher_pipeline(self) -> None:
        """Met en route la file de synthèse, que le TTS soit local ou distant."""
        if self.tts is None or self.speaker is None:
            return
        self.speech = SpeechPipeline(self.tts, self.speaker, cleaner=txt.clean_for_speech)
        self.speech.start()

    # ------------------------------------------------------------------ profils
    def _reglages_profil(self, profil: store.Profil) -> None:
        """Applique les réglages d'un profil : voix, vitesse, langue.

        Au démarrage, le TTS n'existe pas encore : on ne renseigne que la
        configuration, que ``setup_voice`` lira ensuite. En session, on applique
        aussi au TTS vivant.

        La langue passe **avant** la voix : ``set_lang`` reconstruit le pipeline et
        recharge la voix courante, donc changer la voix après évite de la charger
        deux fois.
        """
        if profil.langue:
            self.cfg.lang = profil.langue
            if self.tts:
                self.tts.set_lang(profil.langue)
        if profil.voix:
            self.cfg.voice = profil.voix
            if self.tts:
                self.tts.set_voice(profil.voix)
        if profil.vitesse:
            self.cfg.speed = profil.vitesse
            if self.tts:
                self.tts.speed = profil.vitesse

    def _appliquer_profil(self, nom: str) -> None:
        """Charge un profil en session : prompt système, puis ses réglages."""
        profil = store.lire_profil(nom)
        if not profil.prompt:
            raise ValueError(f"le profil « {nom} » ne contient aucun prompt")
        self._reglages_profil(profil)
        self.cfg.system = profil.prompt
        self.messages[0] = {"role": "system", "content": profil.prompt}
        self.profil = store.nom_fichier(nom)
        print(f"profil « {self.profil} » chargé ({len(profil.prompt)} caractères)")
        if profil.langue:
            print(f"  langue  : {profil.langue}")
        if profil.voix:
            print(f"  voix    : {decrire_voix(profil.voix)}")
        if profil.vitesse:
            print(f"  vitesse : {profil.vitesse}")

    # ------------------------------------------------------------- entrée vocale
    def _assurer_stt(self) -> bool:
        """Charge Whisper au premier besoin, pas au démarrage.

        Une session au clavier ne doit payer ni la seconde de chargement ni les
        ~330 Mo de VRAM du modèle.
        """
        if self.transcripteur is not None and self.transcripteur.pret:
            return True
        if self._stt_indisponible:
            return False

        transcriber = charger_transcripteur(self.cfg)
        if transcriber is None:
            self._stt_indisponible = True
            return False
        self.transcripteur = transcriber
        return True

    def _assurer_ecouteur(self) -> bool:
        if self.ecouteur is not None:
            return True
        micro = Microphone(device=self.cfg.micro_device)
        if not micro.available:
            print(f"Micro  : {micro.erreur or 'entrée audio indisponible'}")
            self._stt_indisponible = True
            return False
        ecouteur = Ecouteur(micro=micro, on_parole=self._debut_de_parole)
        if not ecouteur.vad.pret:
            print(f"Micro  : {ecouteur.vad.erreur}")
            self._stt_indisponible = True
            return False
        self.ecouteur = ecouteur
        return True

    @staticmethod
    def _debut_de_parole() -> None:
        """Marqueur visuel : le VAD vient de détecter le début de la parole."""
        print("  ●", end="", flush=True)

    def dicter(self) -> str | None:
        """Écoute une phrase au micro et la transcrit. ``None`` si rien d'exploitable.

        Lève KeyboardInterrupt si l'utilisateur fait Ctrl+C pendant l'écoute.
        """
        if not self._assurer_stt() or not self._assurer_ecouteur():
            return None

        assert self.ecouteur is not None and self.transcripteur is not None
        print("micro  : parlez… (Ctrl+C pour revenir au clavier)")
        audio = self.ecouteur.ecouter()
        print()  # termine la ligne des « ● »
        if audio is None:
            print("   (rien entendu)")
            return None

        print(f"   {len(audio) / 16000:.1f} s captées — transcription…")
        resultat = self.transcripteur.transcrire(audio)
        if not resultat.texte:
            print(f"   (transcription vide — {resultat.resume()})")
            return None
        print(f"   {resultat.resume()}")
        print(f"vous (voix) › {resultat.texte}")
        return resultat.texte

    def _avis_reseau(self, numero: int, attente: float, raison: str) -> None:
        """Dit qu'on réessaie, et pourquoi.

        Un retry silencieux ressemble à un blocage : l'utilisateur voit le curseur
        immobile et croit que c'est cassé. Ici, il apprend qu'on attend, combien de
        temps, et sur quel échec.
        """
        premiere_ligne = next(
            (l.strip() for l in raison.splitlines() if l.strip()), "échec sans détail"
        )
        print(
            f"\n[réseau] essai {numero} échoué — nouvelle tentative dans {attente:.1f} s",
            flush=True,
        )
        print(f"         ({premiere_ligne[:100]})", flush=True)

    def _avis_voix_manquante(self) -> None:
        """Signale les phrases que la voix n'a pas pu dire — seulement s'il y en a de nouvelles.

        Le texte reste à l'écran, donc rien n'est perdu pour la lecture : ce qui
        manque, c'est le son. Sans cette ligne, l'utilisateur croirait que la voix a
        sauté un passage sans raison.
        """
        attente = self.speech.en_attente if self.speech else 0
        if attente > self._attente_vue:
            print(f"[voix] {attente} phrase(s) non synthétisée(s) — "
                  f"/rejoue quand le GPU est revenu")
            self._attente_vue = attente

    def resolve_model(self) -> bool:
        if self.cfg.model:
            self._mesurer_contexte()
            return True
        try:
            modeles, cible = premier_serveur(self.cfg.cibles, self.cfg.api_key, timeout=10.0)
        except LLMError as exc:
            print(f"\n[ERREUR] {exc}\n")
            print("Astuce : `--probe` teste chaque serveur séparément.")
            return False
        if cible != self.cfg.base_url:
            print(f"Serveur: la cible principale ({self.cfg.base_url}) ne répond pas.")
            print(f"         réponse de {cible}")
            self.serveur_actif = cible
        if not modeles:
            print("[ERREUR] /v1/models ne liste aucun modèle. Passer --model <nom>.")
            return False
        self.cfg.model = modeles[0]
        self.modeles = list(modeles)
        extra = f" ({len(modeles)} disponibles)" if len(modeles) > 1 else ""
        print(f"Modèle : {nom_court(self.cfg.model)}{extra}")
        self._mesurer_contexte()
        return True

    def _mesurer_contexte(self) -> None:
        """Demande au serveur la taille de son contexte, pour compter juste ensuite.

        Sans cette mesure, le client comptait les messages **à l'aveugle** : mesuré, 25
        messages ne pesaient que 1 646 tokens, soit 5 % d'un contexte de 32768. Autrement
        dit, on jetait des messages vingt fois trop tôt, et une conversation dense aurait
        pu déborder sans que rien ne le signale.
        """
        cible = self.serveur_actif or self.cfg.base_url
        if self.cfg.n_ctx:
            print(f"Contexte : {self.cfg.n_ctx} tokens (forcé par la configuration)")
            return
        self.cfg.n_ctx = contexte_du_serveur(cible, self.cfg.api_key)
        if self.cfg.n_ctx:
            print(f"Contexte : {self.cfg.n_ctx} tokens")
        else:
            print("Contexte : le serveur ne l'annonce pas — repli sur l'ancienne limite en messages")

    # ------------------------------------------------------------ complétion
    def _voix_disponibles(self) -> list[str]:
        """Voix Kokoro, récupérées une seule fois (l'appel passe par le réseau)."""
        if self._voix is None:
            try:
                self._voix = list_voices()
            except Exception:
                self._voix = []
            if not self._voix:  # hors ligne : on reste utile avec la voix courante
                self._voix = [self.cfg.voice]
        return self._voix

    def _completeur(self, avant: str) -> list[str]:
        """Candidats pour la touche Tab, selon ce qui est déjà tapé.

        Appelé par l'éditeur, qui ne sait rien de nos commandes.
        """
        if not avant.startswith("/"):
            return []
        if " " not in avant:
            return NOMS_COMMANDES  # on complète le nom de la commande
        commande = avant.partition(" ")[0]
        if commande == "/voice":
            return self._voix_disponibles()
        if commande in ("/load", "/forget"):
            return [c["nom"] for c in store.lister()]
        if commande == "/profil":
            return store.lister_profils() + ["save"]
        if commande == "/tts":
            return ["on", "off"]
        if commande == "/lang":
            return ["a", "b", "e", "f", "h", "i", "j", "p", "z"]
        if commande == "/model":
            return list(self.modeles)
        return []

    # ------------------------------------------------------------------- un tour
    def ask(self, question: str) -> None:
        assert self.cfg.model
        self.messages.append({"role": "user", "content": question})
        self._trim_history()
        # Taille de l'historique telle qu'elle part : le serveur va nous en donner le
        # compte exact en tokens, et c'est sur ce couple qu'on s'ancrera ensuite.
        self._ancrage_messages = len(self.messages)

        connexion = Connexion()
        flux, usage = stream_with_usage(
            base_url=self.cfg.cibles,
            model=self.cfg.model,
            messages=self.messages,
            api_key=self.cfg.api_key,
            temperature=self.cfg.temperature,
            timeout=self.cfg.timeout,
            connexion=connexion,
            on_essai=self._avis_reseau,
        )

        # Le premier morceau ouvre réellement la connexion : c'est là qu'on apprend
        # quel serveur a répondu. On le demande donc avant d'afficher « ia › », pour
        # que la mention de bascule ne tombe pas au milieu de la réponse.
        premier: str | None = None
        try:
            premier = next(flux)
        except StopIteration:
            pass

        if connexion.bascule and connexion.url:
            print(f"[réseau] {self.cfg.base_url} n'a pas répondu → réponse de {connexion.url}")
            self.serveur_actif = connexion.url

        print("ia › ", end="", flush=True)
        tampon = ""
        reponse = ""
        interrompu = False
        erreur = False
        coupee = False

        # Le clavier reste écouté pendant tout l'échange : Échap coupe la voix sans
        # perdre la réponse, et les frappes faites entre-temps ne sont pas jetées.
        with ClavierGeneration() as clavier:
            # `recus` accumule au fil de l'eau plutôt que de compter sur la valeur de
            # retour : si l'utilisateur interrompt en pleine génération, on veut garder ce
            # qui a déjà été lu — c'est ce qu'il a vu à l'écran.
            recus: list[str] = []

            def sur_texte(morceau: str) -> None:
                print(morceau, end="", flush=True)
                recus.append(morceau)

            def surveiller_clavier() -> None:
                """Échap coupe la voix — la génération, elle, continue jusqu'au bout.

                C'est un effet de bord, pas un arrêt : mettre fin au flux ici tronquerait
                la réponse à l'écran, ce que la vérification d'interruption refuse.
                """
                nonlocal coupee
                if clavier.sonder():
                    coupee = True
                    self._couper_voix()

            try:
                morceaux = flux if premier is None else itertools.chain([premier], flux)
                _, tampon, _ = derouler(
                    morceaux,
                    on_texte=sur_texte,
                    on_phrase=self._speak,
                    sur_morceau=surveiller_clavier,
                )
                reponse = "".join(recus)
            except KeyboardInterrupt:
                interrompu = True
            except LLMError as exc:
                erreur = True
                print(f"\n[ERREUR] {exc}")
                self.messages.pop()  # on retire la question pour garder un contexte propre
            finally:
                reponse = "".join(recus) or reponse
                # On ne présume pas du type : `stream_with_usage` annonce un itérateur,
                # mais en pratique c'est un générateur — donc fermable, ce qui libère la
                # connexion HTTP du modèle.
                fermer = getattr(flux, "close", None)
                if fermer is not None:
                    fermer()

            print()  # fin de la ligne de réponse

            if erreur:
                self._couper_voix()
            elif interrompu:
                self._couper_voix()
                coupee = True
                print("[Ctrl+C] génération, synthèse et lecture coupées")
            else:
                if coupee:
                    print("[Échap] voix coupée — la réponse reste à l'écran")
                # Le texte resté sans ponctuation finale part quand même à la voix —
                # sauf si elle vient d'être coupée.
                if tampon.strip() and not coupee:
                    self._speak(tampon.strip())
                if self.speech and not coupee:
                    coupee = self._attendre_voix(clavier)

            # On conserve une réponse partielle : c'est ce que l'utilisateur a lu.
            if reponse.strip() and not erreur:
                self.messages.append({"role": "assistant", "content": reponse})

        self._avis_voix_manquante()
        if usage.prompt_tokens:
            # Compte exact, donné par le serveur pour la liste qu'on vient d'envoyer :
            # il servira d'ancrage aux tours suivants.
            self._ancrage_jetons = usage.prompt_tokens
        self._avis_contexte(usage)

        self.frappes = clavier.tampon
        self._maj_stats(usage, interrompu, coupee)

    def _maj_stats(self, usage, interrompu: bool, coupee: bool) -> None:
        self.last_stats = usage.resume()
        if self.tts and self.tts.audio_s > 0:
            self.last_stats += f" | TTS RTF {self.tts.rtf:.2f}"
        if self.cache is not None and self.cache.hits:
            self.last_stats += f" | réserve {self.cache.hits} reprise(s)"
        if self.cfg.n_ctx:
            self.last_stats += f" | contexte {self._jetons()}/{self.cfg.n_ctx}"
        if interrompu:
            self.last_stats += " | réponse tronquée"
        if coupee:
            self.last_stats += " | voix coupée"
        if self.cfg.show_stats:
            print(f"   · {self.last_stats}")

    def _couper_voix(self) -> None:
        """Vide la file de synthèse, arrête le son, invalide ce qui est en cours."""
        if self.speech:
            self.speech.flush()

    def _attendre_voix(self, clavier: ClavierGeneration) -> bool:
        """Attend la fin de la voix en restant à l'écoute d'Échap.

        L'attente réelle se fait dans un thread : ``wait()`` est la seule source de
        vérité sur « tout est joué » (regarder la file d'attente suffirait à croire
        que c'est fini pendant qu'une phrase est encore en cours de synthèse).
        Retourne True si la voix a été coupée.
        """
        fini = threading.Event()

        def attendre() -> None:
            try:
                self.speech.wait()  # type: ignore[union-attr]
            finally:
                fini.set()

        threading.Thread(target=attendre, name="attente-voix", daemon=True).start()
        while not fini.wait(0.02):
            if clavier.sonder():
                self._couper_voix()
                print("[Échap] voix coupée — la réponse reste à l'écran")
                return True
        return False

    def _speak(self, phrase: str) -> None:
        if self.speech and self.cfg.tts:
            self.speech.say(phrase)

    # ------------------------------------------------------------- contexte
    def _jetons(self) -> int:
        """Tokens que pèsent les messages actuels, aussi juste que possible.

        On part du compte **exact** annoncé par le serveur au tour précédent
        (``usage.prompt_tokens``) et on n'estime que ce qui a été ajouté depuis — deux
        messages, donc une erreur minuscule. Sans ancrage (premier tour, ou messages
        retirés depuis), on estime tout : ~3,5 caractères par token, plus la poignée de
        tokens de gabarit que chaque message ajoute au prompt.
        """
        if self._ancrage_jetons and self._ancrage_messages <= len(self.messages):
            ajoutes = self.messages[self._ancrage_messages :]
            return self._ancrage_jetons + sum(
                _jetons_estimes(m.get("content", "")) for m in ajoutes
            )
        return sum(_jetons_estimes(m.get("content", "")) for m in self.messages) + 4 * len(
            self.messages
        )

    def _limite_jetons(self) -> int:
        """Budget à ne pas dépasser : 80 % du contexte annoncé par le serveur.

        Les 20 % restants sont pour la réponse. On ignore sa longueur avant de l'avoir
        reçue, et un prompt qui remplit exactement le contexte se fait tronquer ou
        refuser par le serveur — au pire moment, après avoir payé tout le calcul.
        """
        return int(self.cfg.n_ctx * 0.8) if self.cfg.n_ctx else 0

    def _prevenir_oubli(self, oublies: int, details: str) -> None:
        """Annonce une fois par vague ce qui sort du contexte.

        Un historique qui rétrécit en silence donne l'impression que le modèle devient
        bête, alors qu'il a simplement cessé de recevoir le début.
        """
        if self._oubli_signale:
            return
        self._oubli_signale = True
        print(f"\n[contexte] {oublies} message(s) ancien(s) ne sont plus envoyés ({details}).")
        print("           /resume pour les compacter en un résumé, au lieu de les perdre.")

    def _trim_history(self) -> None:
        """Garde le message système et autant d'échanges que le contexte le permet.

        On compte en **tokens** dès que le serveur annonce son contexte, et non en
        nombre de messages. L'ancienne limite fixe (24 messages) jetait des messages
        vingt fois trop tôt sur ce serveur, et aurait laissé passer un long échange
        dense sans rien dire.
        """
        limite = self._limite_jetons()
        if not limite:
            # Le serveur n'annonce pas son contexte : ancien comportement, en messages.
            limite_msg = max(2, self.cfg.history_limit)
            if len(self.messages) > limite_msg + 1:
                oublies = len(self.messages) - limite_msg - 1
                self.messages = [self.messages[0]] + self.messages[-limite_msg:]
                self._prevenir_oubli(oublies, f"limite : {limite_msg} messages")
            return

        # On retire par échange complet (question + réponse) : commencer l'historique
        # sur une réponse orpheline est déroutant pour le modèle comme pour le lecteur.
        oublies = 0
        while len(self.messages) > 3 and self._jetons() > limite:
            del self.messages[1:3]
            oublies += 2
            self._ancrage_jetons = 0  # le compte exact ne décrit plus cette liste
        if oublies:
            self._prevenir_oubli(oublies, f"limite : {limite} tokens")

    def _montrer_contexte(self) -> None:
        """Où en est le contexte, et ce que ça implique concrètement.

        C'est la réponse chiffrée à « est-ce qu'il me faut plus de contexte ? » : tant
        que le pourcentage reste bas, monter la taille côté serveur ne changerait rien.
        """
        if not self.cfg.n_ctx:
            print("contexte : le serveur ne l'annonce pas (pas de /props)")
            print(f"           repli sur la limite de {self.cfg.history_limit} messages")
            return
        utilise = self._jetons()
        part = utilise / self.cfg.n_ctx
        print(f"contexte : {utilise} / {self.cfg.n_ctx} tokens ({part:.1%})")
        print(f"           {len(self.messages)} message(s) dans l'historique")
        if self._ancrage_jetons:
            # Preuve que les chiffres ne sont pas devinés : c'est le compte exact que le
            # serveur a rendu au tour précédent, pour l'historique tel qu'il est parti.
            print(
                f"           dernier compte exact du serveur : {self._ancrage_jetons} tokens "
                f"pour {self._ancrage_messages} message(s)"
            )
        print(f"           budget d'envoi : {self._limite_jetons()} tokens (80 % du contexte)")
        if part >= 0.8:
            print("           → proche de la limite : /resume pour compacter")
        else:
            facteur = (self._limite_jetons() - utilise) / max(1, utilise)
            print(f"           → de la place : environ {facteur:.0f} fois la conversation actuelle")

    def _avis_contexte(self, usage) -> None:
        """Prévient quand la conversation approche du contexte du serveur."""
        if not self.cfg.n_ctx:
            return
        utilise = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        if not utilise:
            return
        part = utilise / self.cfg.n_ctx
        if part >= 0.8 and not self._contexte_signale:
            self._contexte_signale = True
            print(
                f"[contexte] {utilise} / {self.cfg.n_ctx} tokens ({part:.0%}) — "
                "/resume pour compacter avant que ça déborde"
            )

    def _compacter_contexte(self) -> bool:
        """Remplace les échanges anciens par un résumé produit par le modèle.

        Ce qu'on garde intact : le prompt système et le dernier échange (question +
        réponse). Tout le reste part dans un résumé qui prend la place des messages
        d'origine. Si le modèle échoue, l'historique n'est **pas** touché — mieux vaut
        un contexte long qu'un contexte perdu.
        """
        anciens = self.messages[1:]
        if len(anciens) <= 2:
            print("[contexte] rien à compacter : la conversation est trop courte")
            return False

        gardes = anciens[-2:]
        a_resumer = anciens[:-2]
        conversation = "\n\n".join(
            f"{'Utilisateur' if m.get('role') == 'user' else 'Assistant'} : {m.get('content', '')}"
            for m in a_resumer
        )
        print(f"[contexte] compactage de {len(a_resumer)} message(s)…")

        # Plafond de longueur. Il borne la durée de l'appel sur un long historique, mais
        # il ne doit pas couper le résumé en pleine phrase : un résumé tronqué perd
        # justement ce qu'on cherche à garder.
        #
        # Mesures sur ce modèle (Ornith-1.5-35B) : sur un échange technique de 1572
        # caractères, il produit un résumé de 1001 caractères (1,6x) et s'arrête au
        # plafond s'il est trop bas — d'où un plancher généreux. Ce modèle n'est pas
        # compressif : mieux vaut un résumé un peu long qu'un résumé amputé.
        plafond = max(512, min(2000, len(conversation) // 6))

        resume = ""
        usage = None
        try:
            flux, usage = stream_with_usage(
                base_url=self.cfg.cibles,
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": PROMPT_RESUME},
                    {"role": "user", "content": conversation},
                ],
                api_key=self.cfg.api_key,
                temperature=0.2,
                timeout=self.cfg.timeout,
                max_tokens=plafond,
            )
            for morceau in flux:
                resume += morceau
        except KeyboardInterrupt:
            print("\n[contexte] compactage interrompu — historique inchangé")
            return False
        except LLMError as exc:
            print(f"[contexte] compactage impossible : {exc}")
            print("           historique inchangé (rien n'a été perdu).")
            return False

        resume = resume.strip()
        if not resume:
            print("[contexte] résumé vide — historique inchangé")
            return False

        self.messages = [
            self.messages[0],
            {"role": "system", "content": f"[Résumé des échanges précédents]\n{resume}"},
            *gardes,
        ]
        self._oubli_signale = False
        self._contexte_signale = False
        # La liste a changé du tout au tout : le compte exact d'avant ne la décrit plus.
        self._ancrage_jetons = 0
        self._ancrage_messages = 0
        print(
            f"[contexte] {len(conversation)} caractères résumés en {len(resume)} ; "
            f"{len(self.messages)} message(s) en contexte."
        )
        if len(resume) >= len(conversation) * 0.9:
            print(
                "           (le résumé n'est pas plus court que le texte d'origine : "
                "il n'y a pas grand-chose à y gagner sur cet échange)"
            )
        elif usage is not None and usage.completion_tokens >= plafond:
            # Ne jamais taire une troncature : le résumé a l'air complet alors qu'il
            # s'arrête au milieu, et c'est tout le début de la conversation qui est en jeu.
            print(
                f"           (ATTENTION : le résumé a atteint son plafond de {plafond} tokens "
                "et s'arrête peut-être en pleine phrase)"
            )
        return True

    # ------------------------------------------------------------ conversations
    def _appliquer(self, conv: store.Conversation) -> None:
        """Remplace l'état de la session par celui d'une conversation chargée."""
        systeme = conv.systeme or self.cfg.system
        sans_systeme = [m for m in conv.messages if m["role"] != "system"]
        self.messages = [{"role": "system", "content": systeme}] + sans_systeme
        self.cfg.system = systeme
        self.conversation = conv.nom
        self.debut = conv.cree

    def reprendre(self) -> None:
        """Recharge la dernière conversation sauvegardée (option --continue)."""
        conv = store.derniere()
        if conv is None:
            print(f"Aucune conversation à reprendre dans {store.dossier()}")
            return
        self._appliquer(conv)
        quand = time.strftime("%d/%m %H:%M", time.localtime(conv.maj))
        print(f"Reprise de « {conv.nom} » — {conv.echanges} messages (dernière fois {quand})")

    # ------------------------------------------------------------------ commandes
    def handle_command(self, ligne: str) -> bool:
        """Retourne False si la session doit se terminer."""
        parts = ligne.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            return False

        if cmd == "/help":
            print(HELP)

        elif cmd == "/reset":
            self.messages = [{"role": "system", "content": self.cfg.system}]
            self._oubli_signale = False
            self._contexte_signale = False
            self._ancrage_jetons = 0
            self._ancrage_messages = 0
            print("historique vidé")

        elif cmd == "/resume":
            self._compacter_contexte()

        elif cmd == "/contexte":
            self._montrer_contexte()

        elif cmd == "/voices":
            for v in list_voices():
                print(" ", v)

        elif cmd == "/device":
            print(self.speaker.describe_devices() if self.speaker else "audio désactivé")

        elif cmd == "/micro":
            if arg.lower() in ("on", "1", "true", "oui"):
                self.mains_libres = True
            elif arg.lower() in ("off", "0", "false", "non"):
                self.mains_libres = False
            else:
                etat = "activé" if self.mains_libres else "désactivé"
                print(f"mode mains libres : {etat}   (usage : /micro on|off)")
                return True
            print(f"mode mains libres {'activé' if self.mains_libres else 'désactivé'}")

        elif cmd == "/tts" and arg:
            self.cfg.tts = arg.lower() in ("on", "1", "true", "oui")
            print(f"voix {'activée' if self.cfg.tts else 'coupée'}")

        elif cmd == "/cache":
            if self.cache is None:
                print("réserve inactive (voix désactivée dans cette session)")
            elif arg.lower() in ("vider", "vide", "clear", "reset"):
                combien = self.cache.vider()
                print(f"réserve vidée : {combien} fichier(s) retiré(s)")
                print("  (les phrases seront resynthétisées au prochain passage)")
            else:
                print(f"réserve : {self.cache.dossier}")
                print(f"  {self.cache.resume()}")
                if self.tts:
                    print(f"  ~{self.tts.temps_economise_s:.2f} s de synthèse évitées "
                          f"(estimation d'après le RTF de la session)")
                print("  vider : /cache vider")

        elif cmd == "/rejoue":
            if not self.speech:
                print("voix indisponible dans cette session")
            elif not self.speech.en_panne:
                print("rien à rejouer : toutes les phrases ont été synthétisées")
            else:
                combien = self.speech.rejouer()
                self._attente_vue = 0
                print(f"{combien} phrase(s) remise(s) en file pour la voix")

        elif cmd == "/voice" and arg:
            if not self.tts:
                print("TTS indisponible")
            else:
                try:
                    # Valider la syntaxe avant de toucher au TTS : une faute de frappe
                    # doit donner un message clair, pas une trace d'exception.
                    analyser_voix(arg)
                except ValueError as exc:
                    print(f"échec : {exc}")
                    print("  syntaxe : nom  |  nom1,nom2  |  nom1:3+nom2:1")
                else:
                    try:
                        self.tts.set_voice(arg)
                        self.cfg.voice = arg
                        print(f"voix = {decrire_voix(arg)}")
                    except Exception as exc:
                        print(f"échec : {exc}")

        elif cmd == "/lang" and arg:
            if not self.tts:
                print("TTS indisponible")
            else:
                try:
                    self.tts.set_lang(arg)
                    self.cfg.lang = arg
                    print(f"langue = {arg}")
                except Exception as exc:
                    print(f"échec : {exc}")

        elif cmd == "/speed" and arg:
            try:
                self.cfg.speed = float(arg)
                if self.tts:
                    self.tts.speed = self.cfg.speed
                print(f"vitesse = {self.cfg.speed}")
            except ValueError:
                print("vitesse invalide")

        elif cmd == "/model" and arg:
            self.cfg.model = arg
            print(f"modèle = {arg}")

        elif cmd == "/system" and arg:
            self.cfg.system = arg
            self.messages[0] = {"role": "system", "content": arg}
            print("prompt système remplacé")

        elif cmd == "/save":
            nom = arg or self.conversation or store.NOM_DEFAUT
            try:
                chemin = store.enregistrer(
                    nom,
                    self.messages,
                    modele=self.cfg.model,
                    systeme=self.cfg.system,
                    cree=self.debut,
                )
            except (ValueError, OSError) as exc:
                print(f"échec de l'enregistrement : {exc}")
            else:
                self.conversation = chemin.stem
                print(f"enregistré « {chemin.stem} » ({len(self.messages) - 1} messages)")
                print(f"  {chemin}")

        elif cmd == "/load":
            if not arg:
                print("usage : /load <nom>   (voir /conversations)")
            else:
                try:
                    conv = store.charger(arg)
                except FileNotFoundError as exc:
                    print(f"échec : {exc}")
                except ValueError as exc:
                    print(f"échec : {exc}")
                else:
                    self._appliquer(conv)
                    if conv.modele:
                        print(f"(conversation enregistrée avec {nom_court(conv.modele)})")
                    print(f"chargé « {conv.nom} » : {conv.echanges} messages")

        elif cmd == "/cherche":
            if not arg.strip():
                print("usage : /cherche <motif>   (cherche dans toutes les conversations)")
            else:
                trouvailles = store.chercher(arg)
                if not trouvailles:
                    print(f"rien trouvé pour « {arg.strip()} »")
                else:
                    print(f"{len(trouvailles)} passage(s) pour « {arg.strip()} » :")
                    for t in trouvailles:
                        quand = time.strftime("%d/%m %H:%M", time.localtime(t.date))
                        print(
                            f"  [{t.conversation}] {quand} · {t.role_lisible()} · msg {t.message}"
                        )
                        print(f"      {t.extrait}")
                    print("  (reprendre une conversation : /load <nom>)")

        elif cmd == "/conversations":
            sauvegardes = store.lister()
            if not sauvegardes:
                print(f"aucune conversation dans {store.dossier()}")
            else:
                print(f"conversations dans {store.dossier()} :")
                for c in sauvegardes:
                    quand = time.strftime("%d/%m %H:%M", time.localtime(c["maj"]))
                    modele = f"  {nom_court(c['modele'])}" if c["modele"] else ""
                    print(f"  {c['nom']:<28} {c['echanges']:>3} messages  {quand}{modele}")

        elif cmd == "/forget":
            if not arg:
                print("usage : /forget <nom>")
            elif store.supprimer(arg):
                print(f"supprimé « {store.nom_fichier(arg)} »")
            else:
                print(f"aucune conversation nommée « {arg} »")

        elif cmd == "/profil":
            sous = arg.split(maxsplit=1)
            if not arg:
                profils = store.lister_profils()
                dossier_p = store.dossier_profils()
                if not profils:
                    print(f"aucun profil dans {dossier_p}")
                    print("  créer le courant avec : /profil save <nom>")
                else:
                    print(f"profils dans {dossier_p} :")
                    for nom in profils:
                        marque = "   ← actif" if nom == self.profil else ""
                        print(f"  {nom}{marque}")

            elif sous[0] == "save":
                nom = sous[1].strip() if len(sous) > 1 else ""
                if not nom:
                    print("usage : /profil save <nom>")
                else:
                    # On n'écrit un réglage que s'il s'écarte du défaut : un profil qui
                    # ne change pas la voix reste un simple fichier de prompt, comme
                    # avant la v0.5. C'est ce qui rend la relecture du fichier utile.
                    voix_ecrite = self.cfg.voice if self.cfg.voice != DEFAULT_VOICE else None
                    try:
                        chemin = store.enregistrer_profil(
                            nom,
                            self.cfg.system,
                            voix=voix_ecrite,
                            vitesse=(
                                self.cfg.speed if self.cfg.speed != DEFAULT_SPEED else None
                            ),
                            langue=(
                                self.cfg.lang if self.cfg.lang != DEFAULT_LANG else None
                            ),
                        )
                    except (ValueError, OSError) as exc:
                        print(f"échec : {exc}")
                    else:
                        self.profil = chemin.stem
                        print(f"profil « {chemin.stem} » enregistré")
                        print(f"  {chemin}")
                        if voix_ecrite:
                            print(f"  voix    : {decrire_voix(voix_ecrite)}")

            else:
                try:
                    self._appliquer_profil(arg)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"échec : {exc}")

        elif cmd == "/export":
            nom_fichier = arg or f"conversation-{time.strftime('%Y%m%d-%H%M')}.md"
            if not nom_fichier.endswith(".md"):
                nom_fichier += ".md"
            try:
                chemin = store.exporter_markdown(
                    self.messages,
                    Path(nom_fichier),
                    titre=self.conversation or "Conversation",
                    modele=nom_court(self.cfg.model),
                    voix=self.cfg.voice if self.cfg.tts else "",
                )
            except OSError as exc:
                print(f"échec de l'export : {exc}")
            else:
                print(f"exporté : {chemin}  ({len(self.messages) - 1} messages)")

        elif cmd == "/stats":
            print(self.last_stats or "aucune mesure pour l'instant")

        elif cmd == "/debug":
            self.cfg.show_stats = not self.cfg.show_stats
            print(f"stats {'activées' if self.cfg.show_stats else 'désactivées'}")

        else:
            print(f"commande inconnue : {cmd} — /help pour la liste")

        return True

    # --------------------------------------------------------------------- boucle
    def run(self, reprendre: bool = False) -> int:
        cfg = self.cfg
        print(BANNER)
        print(f"voicechat {__version__}  ·  serveur {cfg.base_url}")

        if not self.resolve_model():
            return 2

        # Réglages propres à ce modèle, s'ils existent dans le fichier de configuration.
        # On le fait ici : le modèle vient d'être résolu, et la voix n'est pas encore
        # chargée — le TTS démarrera donc directement avec la bonne.
        self.cfg, ajustes = self.cfg.pour_modele(self.cfg.model)
        if ajustes:
            details = ", ".join(f"{c}={v}" for c, v in sorted(ajustes.items()))
            print(f"Modèle : réglages du fichier de configuration — {details}")
            if "system" in ajustes:
                self.messages[0] = {"role": "system", "content": self.cfg.system}

        if self.profil:
            print(f"Profil : {self.profil}")

        if reprendre:
            self.reprendre()

        self.setup_voice()

        print()
        if self.mains_libres:
            print("Micro  : mode mains libres — parlez après chaque réponse.")
            print("         Ctrl+C pendant l'écoute rend la main au clavier.")
        print("Tapez votre message, ou collez un texte — Entrée pour l'envoyer.")
        print("/help pour l'aide, /quit pour sortir.")
        if self.conversation:
            print(f"conversation : « {self.conversation} »  (/save pour l'enregistrer)")
        print("─" * 60)

        while True:
            ligne: str | None
            if self.mains_libres:
                # Mains libres : c'est le micro qui fournit le message.
                try:
                    ligne = self.dicter()
                except KeyboardInterrupt:
                    self.mains_libres = False
                    print("\n[micro] mains libres désactivé — retour au clavier")
                    continue
                if not ligne:
                    continue  # rien entendu : on réécoute
            else:
                try:
                    initial, self.frappes = self.frappes, ""
                    ligne = self.editor.read(initial=initial)
                except KeyboardInterrupt:
                    print()
                    break
                if ligne is None:  # Ctrl+D ou fin d'entrée
                    print()
                    break
                if not ligne:
                    continue

            if ligne.startswith("/ecoute"):
                try:
                    dicte = self.dicter()
                except KeyboardInterrupt:
                    print("\n[écoute] interrompue")
                    continue
                if dicte:
                    self.ask(dicte)
                continue

            if ligne.startswith("/"):
                if not self.handle_command(ligne):
                    break
                continue

            self.ask(ligne)

        print("à bientôt ★")
        if self.ecouteur:
            self.ecouteur.micro.fermer()
        if self.speech:
            self.speech.close()
        if self.speaker:
            self.speaker.close()
        return 0


# ------------------------------------------------------------------------ entrée
def do_web(cfg: Config, port: int | None = None) -> int:
    """Sert l'interface web : la même conversation, dans un navigateur.

    Le serveur tient **une** `ChatSession` : historique, profils, réserve d'audio et
    mesure du contexte sont donc ceux de la console. La synthèse reste ici aussi, ce qui
    permet d'ouvrir la page depuis un téléphone qui n'a ni GPU ni Kokoro.
    """
    from .web import ServeurWeb

    session = ChatSession(cfg)
    if not session.resolve_model():
        return 2
    session.cfg, _ = session.cfg.pour_modele(session.cfg.model)
    # Pas de pipeline local : la machine ne doit pas se mettre à parler pendant que le
    # navigateur joue déjà le son.
    session.setup_voice(avec_pipeline=False)
    if session.tts is None:
        print("[ERREUR] la voix doit être disponible : c'est elle qui fournit l'audio")
        return 2

    try:
        serveur = ServeurWeb(session.tts, session, port or PORT_WEB)
        url = serveur.demarrer()
    except OSError as exc:
        print(f"[ERREUR] port indisponible : {exc}")
        return 2

    print(f"Page    : {url}")
    print(f"          en local : {serveur.url_local}")
    print(f"Modèle  : {nom_court(session.cfg.model)}")
    if session.cfg.n_ctx:
        print(f"Contexte: {session.cfg.n_ctx} tokens")
    print()
    print("Ouvrir l'adresse depuis n'importe quelle machine du tailnet.")
    print("Pour HTTPS (micro du navigateur, installation sur téléphone) :")
    print(f"  sudo tailscale serve --bg {serveur.port}")
    print()
    print("Ctrl+C pour arrêter.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nservice arrêté.")
    finally:
        serveur.arreter()
    return 0


def do_serveur_stt(cfg: Config, port: int | None = None) -> int:
    """Tient le service de transcription pour d'autres postes, jusqu'à Ctrl+C.

    Le pendant du serveur TTS, dans l'autre sens : le client envoie de l'**audio**, le
    service rend du **texte**. Un client léger — téléphone, navigateur — peut donc parler
    sans embarquer Whisper, ce qui est tout l'intérêt : c'est Whisper qui coûte cher, pas
    le fait de parler.
    """
    from .stt_distant import ServeurSTT

    print(f"Serveur STT : chargement de whisper « {cfg.stt_modele} »…")
    transcriber = Transcriber(
        modele=cfg.stt_modele, device=cfg.stt_device, langue=cfg.stt_langue
    )
    if not transcriber.charger():
        print(f"[ERREUR] {transcriber.erreur}")
        return 2
    print(f"Transcription : {transcriber.device} ({transcriber.compute_reel})")

    port_stt = port or PORT_STT
    try:
        serveur = ServeurSTT(transcriber, port_stt)
        url = serveur.demarrer()
    except OSError as exc:
        print(f"[ERREUR] port {port_stt} indisponible : {exc}")
        return 2

    print(f"Écoute      : {url}")
    print(f"                GET  {url}/sante      (état)")
    print(f"                POST {url}/transcris  (audio → texte)")
    print()
    print("Depuis un autre poste :")
    print(f"  voicechat --transcrire enregistrement.wav --stt-distant {url}")
    print(f"  voicechat --micro --stt-distant {url}")
    print()
    print("Ctrl+C pour arrêter le service.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nservice arrêté.")
    finally:
        serveur.arreter()
    return 0


def do_serveur_tts(cfg: Config, port: int | None = None) -> int:
    """Tient le service TTS pour d'autres postes, jusqu'à Ctrl+C.

    Le poste léger n'a alors besoin que de ``--tts-distant`` : ni torch, ni Kokoro,
    ni GPU chez lui. La réserve d'audio vit ici, donc une phrase déjà dite est
    resservie à n'importe quel client sans repasser par le GPU.
    """
    from .distant import ServeurTTS

    print(f"Serveur TTS : chargement de Kokoro « {decrire_voix(cfg.voice)} »…")
    cache = CacheAudio(actif=cfg.cache)
    tts = KokoroTTS(
        voice=cfg.voice, lang_code=cfg.lang, speed=cfg.speed,
        device=cfg.device, cache=cache,
    )
    if not tts.load():
        print(f"[ERREUR] {tts.load_error}")
        return 2
    print(f"GPU         : {tts.device_reason} → device={tts.device}")
    print(f"Réserve     : {cache.dossier}")

    port_tts = port or PORT_DEFAUT
    try:
        serveur = ServeurTTS(tts, port_tts)
        url = serveur.demarrer()
    except OSError as exc:
        print(f"[ERREUR] port {port_tts} indisponible : {exc}")
        return 2

    print(f"Écoute      : {url}")
    print(f"                GET  {url}/sante   (état)")
    print(f"                POST {url}/parle   (texte → WAV)")
    print()
    print("Depuis un autre poste :")
    print(f"  voicechat --tts-distant {url}")
    print()
    print("Ctrl+C pour arrêter le service.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nservice arrêté.")
    finally:
        serveur.arreter()
    return 0


def do_dire(cfg: Config, texte: str) -> int:
    """Lit un texte à voix haute, sans passer par le LLM.

    Sert à deux choses : comparer deux voix côte à côte sans dépendre du serveur LLM
    (``--voice ff_siwis:3+ef_dora:1 --dire "Bonjour tout le monde."``), et se faire lire
    un document. Le texte passe par le même nettoyage et la même file que les réponses,
    donc ce qu'on entend ici est exactement ce qu'on entendrait d'une réponse.
    """
    if not texte.strip():
        print("[ERREUR] rien à lire")
        return 2

    session = ChatSession(cfg)
    session.setup_voice()
    if session.tts is None or session.speech is None:
        print("[ERREUR] voix indisponible (--no-tts ?)")
        return 2

    # Le découpeur attend un flux : pour un texte complet, le dernier fragment doit être
    # lu aussi, sinon le fichier s'arrête à la dernière ponctuation forte.
    phrases, reste = txt.split_sentences(texte)
    if reste.strip():
        phrases.append(reste)
    if not phrases:
        print("[ERREUR] rien à lire")
        return 2

    print(f"Lecture de {len(phrases)} phrase(s)…")
    try:
        for phrase in phrases:
            session.speech.say(phrase)
        session.speech.wait()
    except KeyboardInterrupt:
        print("\nlecture interrompue")
    finally:
        session.speech.close()
        if session.speaker:
            session.speaker.close()
    return 0


def do_question(cfg: Config, question: str) -> int:
    """Pose une seule question, affiche la réponse, puis sort.

    C'est le mode scriptable : un cron peut faire annoncer un rappel à voix haute, et un
    autre programme peut appeler voicechat sans lui offrir de terminal. Le clavier en mode
    brut de `ask()` sait déjà se taire quand l'entrée n'est pas un terminal — vérifié,
    sinon ce mode aurait planté sous cron.
    """
    session = ChatSession(cfg)
    if not session.resolve_model():
        return 2
    session.cfg, _ = session.cfg.pour_modele(session.cfg.model)
    session.setup_voice()
    try:
        session.ask(question)
    except KeyboardInterrupt:
        print("\ninterrompu")
        return 130
    finally:
        if session.speech:
            session.speech.wait()  # ne pas couper la voix au milieu d'une phrase
            session.speech.close()
        if session.speaker:
            session.speaker.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.list_voices:
        return do_list_voices()
    if args.transcrire:
        return do_transcrire(cfg, args.transcrire)
    if args.dire is not None:
        return do_dire(cfg, args.dire)
    if args.dire_fichier:
        try:
            texte = Path(args.dire_fichier).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[ERREUR] {args.dire_fichier} : {exc}")
            return 2
        return do_dire(cfg, texte)
    if args.question is not None:
        return do_question(cfg, args.question)
    if args.diag_micro:
        return do_diag_micro(cfg)
    if args.serveur_tts:
        return do_serveur_tts(cfg, args.port)
    if args.serveur_stt:
        return do_serveur_stt(cfg, args.port)
    if args.web:
        return do_web(cfg, args.port)
    if args.probe:
        return do_probe(cfg)

    # Aperçu utile avant tout chargement de modèle.
    _, description = resolve_device(cfg.device)
    print(f"[matériel] {description}")

    return ChatSession(cfg).run(reprendre=args.reprendre)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
