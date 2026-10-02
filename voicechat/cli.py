"""Boucle de chat en console : orchestre LLM (streaming) → phrases → voix."""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__, text as txt
from .audio import Speaker
from .config import Config
from .editor import LineEditor
from .llm import LLMError, list_models, stream_with_usage
from .tts import KokoroTTS, SpeechPipeline, list_voices, resolve_device

BANNER = r"""
 __     __    _         _____ _           _
 \ \   / /__ (_) ___ ___|_   _| |__   __| | __ _
  \ \ / / _ \| |/ __/ _ \| | | '_ \ / _` |/ _` |
   \ V / (_) | | (_|  __/| | | | | | (_| | (_| |
    \_/ \___/|_|\___\___||_| |_| |_|\__,_|\__,_|
"""

HELP = """Commandes disponibles :
  /help              cette aide
  /quit  /exit  /q   quitter
  /reset             vide l'historique de conversation
  /voice <nom>       change la voix Kokoro (ex. /voice af_heart)
  /lang <code>       change la langue (a en, b en-GB, f fr, e es, i it, p pt, j ja, z zh, h hi)
  /speed <x>         vitesse de lecture (ex. /speed 1.15)
  /tts on|off        active ou coupe la voix
  /model <nom>       change de modèle pour les tours suivants
  /system <texte>    remplace le prompt système
  /voices            liste les voix Kokoro (nécessite Hugging Face)
  /device            liste les sorties audio détectées
  /stats             dernières mesures (latence, débit, RTF du TTS)
  /debug             bascule l'affichage des stats à chaque tour

Clavier :
  Coller un texte multi-lignes : il est conservé tel quel, rien n'est envoyé
                                 tant que vous n'avez pas frappé Entrée
  ⏎ Entrée        envoyer le message
  ↑ / ↓            historique des messages
  ← / →            déplacer le curseur · Ctrl+U effacer · Ctrl+W effacer le mot
  Ctrl+C           effacer le brouillon (sur ligne vide : quitter)
  Ctrl+D           quitter"""



# --------------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voicechat",
        description="Chat console avec un LLM local, réponses synthétisées par Kokoro.",
    )
    p.add_argument("--base-url", help="racine de l'API OpenAI-compatible (ex. http://hote:8080/v1)")
    p.add_argument("--model", help="nom du modèle (défaut : détecté via /v1/models)")
    p.add_argument("--api-key", help="jeton Bearer si le serveur en exige un")
    p.add_argument("--system", help="prompt système")
    p.add_argument("--voice", help="voix Kokoro (ex. ff_siwis, af_heart)")
    p.add_argument("--lang", help="code langue Kokoro (a=en, b=en-GB, f=fr, e=es, i=it, p=pt, j=ja, z=zh, h=hi)")
    p.add_argument("--speed", type=float, help="vitesse de lecture (1.0 = normal)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], help="périphérique de synthèse")
    p.add_argument("--output-device", help="index ou nom du périphérique de sortie audio")
    p.add_argument("--no-tts", action="store_true", help="désactiver complètement la voix")
    p.add_argument("--temperature", type=float, help="température d'échantillonnage")
    p.add_argument("--timeout", type=float, help="délai max en secondes par requête")
    p.add_argument("--probe", action="store_true", help="tester la joignabilité du serveur et sortir")
    p.add_argument("--list-voices", action="store_true", help="lister les voix Kokoro et sortir")
    p.add_argument("--debug", action="store_true", help="afficher les statistiques à chaque tour")
    p.add_argument("--version", action="version", version=f"voicechat {__version__}")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    out_dev = args.output_device
    if out_dev is not None:
        try:
            out_dev = int(out_dev)
        except (TypeError, ValueError):
            pass  # nom de périphérique : sounddevice accepte aussi une chaîne
    return cfg.with_overrides(
        base_url=args.base_url.rstrip("/") if args.base_url else None,
        model=args.model,
        api_key=args.api_key,
        system=args.system,
        voice=args.voice,
        lang=args.lang,
        speed=args.speed,
        device=args.device,
        output_device=out_dev,
        tts=False if args.no_tts else None,
        temperature=args.temperature,
        timeout=args.timeout,
        show_stats=True if args.debug else None,
    )


# ----------------------------------------------------------------------- modes courts
def do_probe(cfg: Config) -> int:
    print(f"Serveur   : {cfg.base_url}")
    print(f"Test      : GET {cfg.models_url}")
    t0 = time.monotonic()
    try:
        modeles = list_models(cfg.base_url, cfg.api_key, timeout=10.0)
    except LLMError as exc:
        print(f"ÉCHEC après {time.monotonic() - t0:.1f} s\n  {exc}")
        return 2
    print(f"OK en {time.monotonic() - t0:.2f} s — {len(modeles)} modèle(s)")
    for nom in modeles:
        print(f"  • {nom}")
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
    return 0


# --------------------------------------------------------------------- la session
class ChatSession:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.messages: list[dict] = [{"role": "system", "content": cfg.system}]
        self.tts: KokoroTTS | None = None
        self.speaker: Speaker | None = None
        self.speech: SpeechPipeline | None = None
        self.editor = LineEditor()
        self.last_stats = ""

    # ------------------------------------------------------------------ démarrage
    def setup_voice(self) -> None:
        cfg = self.cfg
        if not cfg.tts:
            print("Voix   : désactivée (--no-tts)")
            return

        self.speaker = Speaker(samplerate=24000, device=cfg.output_device)  # type: ignore[arg-type]
        if not self.speaker.available:
            print("Voix   : sounddevice indisponible → chat muet")
            self.cfg.tts = False
            return
        self.speaker.start()

        print(f"Voix   : chargement de Kokoro « {cfg.voice} »…")
        self.tts = KokoroTTS(voice=cfg.voice, lang_code=cfg.lang, speed=cfg.speed, device=cfg.device)
        if not self.tts.load():
            print(f"Voix   : INDISPONIBLE — {self.tts.load_error}")
            print("         (le chat continue en texte ; voir README §8)")
            self.cfg.tts = False
            self.tts = None
            return

        print(f"GPU    : {self.tts.device_reason} → device={self.tts.device}")
        self.speech = SpeechPipeline(self.tts, self.speaker, cleaner=txt.clean_for_speech)
        self.speech.start()

    def resolve_model(self) -> bool:
        if self.cfg.model:
            return True
        try:
            modeles = list_models(self.cfg.base_url, self.cfg.api_key, timeout=10.0)
        except LLMError as exc:
            print(f"\n[ERREUR] {exc}\n")
            print("Astuce : `--probe` refait ce test seul, `--no-tts` n'y change rien.")
            return False
        if not modeles:
            print("[ERREUR] /v1/models ne liste aucun modèle. Passer --model <nom>.")
            return False
        self.cfg.model = modeles[0]
        print(f"Modèle : {self.cfg.model} (détecté, {len(modeles)} disponible(s))")
        return True

    # ------------------------------------------------------------------- un tour
    def ask(self, question: str) -> None:
        assert self.cfg.model
        self.messages.append({"role": "user", "content": question})
        self._trim_history()

        flux, usage = stream_with_usage(
            base_url=self.cfg.base_url,
            model=self.cfg.model,
            messages=self.messages,
            api_key=self.cfg.api_key,
            temperature=self.cfg.temperature,
            timeout=self.cfg.timeout,
        )

        print("ia › ", end="", flush=True)
        tampon = ""
        reponse = ""
        interrompu = False
        t0 = time.monotonic()
        try:
            for morceau in flux:
                print(morceau, end="", flush=True)
                reponse += morceau
                tampon += morceau
                phrases, tampon = txt.split_sentences(tampon)
                for phrase in phrases:
                    self._speak(phrase)
        except KeyboardInterrupt:
            interrompu = True
            if self.speech:
                self.speech.flush()
            print("\n[interrompu par Ctrl+C]")
        except LLMError as exc:
            print(f"\n[ERREUR] {exc}")
            self.messages.pop()  # on retire la question pour garder un contexte propre
            if self.speech:
                self.speech.flush()
            return
        finally:
            print()

        if tampon.strip():
            self._speak(tampon.strip())
        if reponse.strip():
            self.messages.append({"role": "assistant", "content": reponse})

        if self.speech:
            self.speech.wait()  # on laisse finir la voix avant de reprendre la main

        self.last_stats = (
            f"1er token {usage.first_token_s:.2f} s | {usage.chars} car. en "
            f"{usage.total_s or (time.monotonic() - t0):.2f} s "
            f"({usage.chars_per_s:.1f} car/s)"
        )
        if self.tts and self.tts.audio_s > 0:
            self.last_stats += f" | TTS RTF {self.tts.rtf:.2f}"
        if interrompu:
            self.last_stats += " | réponse tronquée"
        if self.cfg.show_stats:
            print(f"   · {self.last_stats}")

    def _speak(self, phrase: str) -> None:
        if self.speech and self.cfg.tts:
            self.speech.say(phrase)

    def _trim_history(self) -> None:
        """Garde le message système + les N derniers messages."""
        limite = max(2, self.cfg.history_limit)
        if len(self.messages) > limite + 1:
            self.messages = [self.messages[0]] + self.messages[-limite:]

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
            print("historique vidé")

        elif cmd == "/voices":
            for v in list_voices():
                print(" ", v)

        elif cmd == "/device":
            print(self.speaker.describe_devices() if self.speaker else "audio désactivé")

        elif cmd == "/tts" and arg:
            self.cfg.tts = arg.lower() in ("on", "1", "true", "oui")
            print(f"voix {'activée' if self.cfg.tts else 'coupée'}")

        elif cmd == "/voice" and arg:
            if not self.tts:
                print("TTS indisponible")
            else:
                try:
                    self.tts.set_voice(arg)
                    self.cfg.voice = arg
                    print(f"voix = {arg}")
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

        elif cmd == "/stats":
            print(self.last_stats or "aucune mesure pour l'instant")

        elif cmd == "/debug":
            self.cfg.show_stats = not self.cfg.show_stats
            print(f"stats {'activées' if self.cfg.show_stats else 'désactivées'}")

        else:
            print(f"commande inconnue : {cmd} — /help pour la liste")

        return True

    # --------------------------------------------------------------------- boucle
    def run(self) -> int:
        cfg = self.cfg
        print(BANNER)
        print(f"voicechat {__version__}  ·  serveur {cfg.base_url}")

        if not self.resolve_model():
            return 2

        self.setup_voice()

        print()
        print("Tapez votre message, ou collez un texte — Entrée pour l'envoyer.")
        print("/help pour l'aide, /quit pour sortir.")
        print("─" * 60)

        while True:
            try:
                ligne = self.editor.read()
            except KeyboardInterrupt:
                print()
                break
            if ligne is None:  # Ctrl+D ou fin d'entrée
                print()
                break
            if not ligne:
                continue
            if ligne.startswith("/"):
                if not self.handle_command(ligne):
                    break
                continue

            self.ask(ligne)

        print("à bientôt ★")
        if self.speech:
            self.speech.close()
        if self.speaker:
            self.speaker.close()
        return 0


# ------------------------------------------------------------------------ entrée
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.list_voices:
        return do_list_voices()
    if args.probe:
        return do_probe(cfg)

    # Aperçu utile avant tout chargement de modèle.
    _, description = resolve_device(cfg.device)
    print(f"[matériel] {description}")

    return ChatSession(cfg).run()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
