"""Interface web : la même conversation, dans un navigateur.

Ce module ne réinvente presque rien — c'est délibéré :

* ``POST /parle`` et ``GET /sante`` viennent du serveur TTS de la v1.0, testé et éprouvé
  (``distant.py``). Le navigateur reçoit donc exactement le même WAV que la console ;
* la découpe des phrases avant la voix est celle de ``tour.derouler``, partagée avec la
  console : si les deux divergeaient, l'écran et le son ne diraient pas la même chose ;
* l'historique, les profils, la réserve d'audio et la mesure du contexte sont ceux de
  ``ChatSession``. Le serveur web ne tient aucun état à lui.

Trois routes, donc, en plus de celles héritées :

    GET  /              la page (un seul fichier, ni framework ni étape de build)
    GET  /flux?q=…      le texte au fil de l'eau, en SSE
    GET  /etat          quel modèle, quelle taille de contexte
    POST /reset         nouvelle conversation

**Pourquoi SSE et pas WebSocket** : le texte va dans un seul sens, et SSE est un simple
GET que le navigateur sait relire tout seul après une coupure. Le WebSocket deviendra
utile le jour où le micro entrera dans la boucle — pas avant, et il s'ajoutera alors sans
rien réécrire de ce qui est ici.

**Ce qui n'est pas partagé, et qu'il faudra réunir** : l'orchestration d'un tour (empiler
la question, appeler le modèle, empiler la réponse, mesurer) est écrite ici une seconde
fois, en plus de ``ChatSession.ask``. C'est assumé pour cette première tranche — le
risque de casser la console en la remaniant était plus grand que le gain — mais c'est
une dette : la prochaine occasion doit faire descendre cette logique dans ``ChatSession``
pour qu'il n'en existe qu'une version.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from . import distant
from .config import nom_court
from .llm import LLMError, stream_with_usage
from .stt_distant import RoutesTranscription
from .tour import derouler

if TYPE_CHECKING:  # évite un cycle à l'exécution : cli.py importe ce module
    from .cli import ChatSession

PORT_WEB = 8091
PAGE = Path(__file__).with_name("web_page.html")


class _HandlerWeb(distant._Handler, RoutesTranscription):
    """Routes du serveur web. La session et le synthétiseur sont posés par ``ServirWeb``.

    Il tient trois rôles : servir la page, porter la conversation (`/flux`, `/reset`), et
    — depuis la v1.2 — **transcrire** (`/transcris`, hérité de ``RoutesTranscription``).
    C'est la même adresse pour parler et pour écouter : un client léger n'en connaît
    qu'une.
    """

    session: "ChatSession | None" = None
    # Un seul tour à la fois : deux tours simultanés se marcheraient dessus dans le même
    # historique. Nom **distinct** de `verrou`, qui est hérité de `distant._Handler` et
    # sert à sérialiser les appels au GPU : les confondre faisait que `/parle`, appelé par
    # la page pendant la génération, attendait la **fin du tour** avant de synthétiser la
    # première phrase. Le son n'arrivait donc qu'une fois la réponse entière affichée —
    # exactement l'inverse du but.
    verrou_tour = threading.Lock()

    # Le transcripteur n'est **pas** chargé au démarrage : Whisper coûte de la VRAM et une
    # quinzaine de secondes, et beaucoup de visiteurs se servent de la page sans jamais
    # parler. Il arrive à la première demande de transcription. `charger_transcription`
    # est posé par ``ServirWeb``, qui seul connaît la configuration.
    # Volontairement `Any` et non `Callable` : annoté `Callable`, Pyright le prend pour
    # une **méthode** et refuse de le lire sur la classe (il attend un `self`). Or c'est
    # bien une fonction posée de l'extérieur, par ``ServirWeb``.
    charger_transcription: Any = None  # () -> Transcription | None
    verrou_chargement = threading.Lock()

    @classmethod
    def _assurer_transcription(cls) -> object | None:
        """Charge Whisper à la première demande, une seule fois même à plusieurs.

        Deux requêtes simultanées ne doivent pas charger deux Whisper : le second
        attendrait pour rien, et le GPU n'en voudrait pas deux de toute façon.
        """
        if cls.transcriber is not None:
            return cls.transcriber
        if cls.charger_transcription is None:
            return None
        with cls.verrou_chargement:
            if cls.transcriber is None:  # re-test : un autre fil a pu charger entre-temps
                cls.transcriber = cls.charger_transcription()
            return cls.transcriber

    def transcrire(self) -> None:
        """``POST /transcris`` : charge Whisper au besoin, puis transcrit."""
        if type(self)._assurer_transcription() is None:
            self._erreur(503, "transcription indisponible : Whisper n'a pas pu être chargé")
            return
        self.servir_transcris()

    # ------------------------------------------------------------------ utilitaires
    def _evenement(self, nom: str, donnees: dict) -> None:
        """Écrit un évènement SSE et le pousse immédiatement au client.

        Le flush est indispensable : sans lui, le texte resterait dans le tampon et
        n'arriverait qu'à la fin — c'est-à-dire jamais en « temps réel ».
        """
        corps = f"event: {nom}\ndata: {json.dumps(donnees, ensure_ascii=False)}\n\n"
        self.wfile.write(corps.encode("utf-8"))
        self.wfile.flush()

    def _entetes_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Pas de Content-Length : la réponse se termine à la fermeture de la connexion.
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    # ----------------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802 (nom imposé par http.server)
        decoupe = urlsplit(self.path)
        chemin = decoupe.path.rstrip("/") or "/"
        if chemin == "/":
            self._page()
            return
        if chemin == "/etat":
            self._etat()
            return
        if chemin == "/flux":
            self._flux(decoupe.query)
            return
        super().do_GET()  # /sante, hérité du serveur TTS

    def do_POST(self) -> None:  # noqa: N802
        chemin = urlsplit(self.path).path.rstrip("/")
        if chemin == "/reset":
            self._reset()
            return
        if chemin == "/transcris":
            self.transcrire()
            return
        super().do_POST()  # /parle, hérité du serveur TTS

    # ---------------------------------------------------------------------- contenus
    def _page(self) -> None:
        try:
            contenu = PAGE.read_bytes()
        except OSError as exc:
            self._erreur(500, f"page introuvable ({exc})")
            return
        self._envoyer(200, contenu, "text/html")

    def _etat(self) -> None:
        session = type(self).session
        if session is None:
            self._erreur(503, "session indisponible")
            return
        tts = type(self).tts
        donnees = {
            "modele": nom_court(session.cfg.model),
            "archives": len(session.messages),
            "voix": getattr(tts, "voice", ""),
            "contexte": None,
            # Ce que la page peut savoir du micro **avant** d'enregistrer : le service est
            # capable de transcrire (le chargeur existe), et Whisper est-il déjà en
            # mémoire. Sans ça, le premier « tenir pour parler » échouerait sans explication
            # pendant la quinzaine de secondes que met Whisper à se charger.
            "transcription": {
                **self.etat_transcription(),
                "disponible": type(self).charger_transcription is not None,
            },
        }
        if session.cfg.n_ctx:
            donnees["contexte"] = {
                "utilise": session._jetons(),
                "total": session.cfg.n_ctx,
                "budget": session._limite_jetons(),
            }
        self._envoyer(
            200, json.dumps(donnees, ensure_ascii=False).encode("utf-8"), "application/json"
        )

    def _reset(self) -> None:
        session = type(self).session
        if session is None:
            self._erreur(503, "session indisponible")
            return
        with type(self).verrou_tour:
            session.messages = [{"role": "system", "content": session.cfg.system}]
            session._oubli_signale = False
            session._contexte_signale = False
            session._ancrage_jetons = 0
            session._ancrage_messages = 0
        self._envoyer(200, b'{"ok": true}', "application/json")

    # ------------------------------------------------------------------- le tour
    def _flux(self, requete: str) -> None:
        """Le cœur : une question en paramètre, du texte puis de l'audio en retour."""
        session = type(self).session
        cls = type(self)

        question = (parse_qs(requete).get("q") or [""])[0].strip()

        # Les en-têtes partent **avant** de savoir si on pourra répondre : un EventSource
        # de navigateur ne lit le corps que sur une réponse 200, donc un 409 ne lui
        # donnerait qu'un « connexion perdue ». On dit l'erreur dans le flux.
        self._entetes_sse()

        if session is None:
            self._evenement("erreur", {"message": "session indisponible"})
            return
        if not question:
            self._evenement("erreur", {"message": "question vide"})
            return
        if not cls.verrou_tour.acquire(blocking=False):
            self._evenement("erreur", {"message": "une réponse est déjà en cours"})
            return

        flux = None
        try:
            session.messages.append({"role": "user", "content": question})
            session._trim_history()
            session._ancrage_messages = len(session.messages)

            self._evenement(
                "debut",
                {
                    "modele": nom_court(session.cfg.model),
                    "contexte": (
                        {"utilise": session._jetons(), "total": session.cfg.n_ctx}
                        if session.cfg.n_ctx
                        else None
                    ),
                },
            )

            flux, usage = stream_with_usage(
                base_url=session.cfg.cibles,
                model=session.cfg.model,
                messages=session.messages,
                api_key=session.cfg.api_key,
                temperature=session.cfg.temperature,
                timeout=session.cfg.timeout,
            )

            debut = time.monotonic()
            reponse, reste, _ = derouler(
                flux,
                on_texte=lambda morceau: self._evenement("texte", {"t": morceau}),
                on_phrase=lambda phrase: self._evenement("phrase", {"t": phrase}),
            )
            secondes = time.monotonic() - debut

            # Le dernier fragment, sans ponctuation finale, se dit quand même : c'est la
            # fin de la réponse, la perdre ferait une phrase tronquée à l'oreille.
            if reste.strip():
                self._evenement("phrase", {"t": reste.strip()})

            if reponse.strip():
                session.messages.append({"role": "assistant", "content": reponse})
            if usage.prompt_tokens:
                session._ancrage_jetons = usage.prompt_tokens

            self._evenement(
                "fin",
                {
                    "tokens": usage.completion_tokens or usage.chars,
                    "secondes": secondes,
                    "tok_s": usage.tok_s,
                    "contexte": (
                        {"utilise": session._jetons(), "total": session.cfg.n_ctx}
                        if session.cfg.n_ctx
                        else None
                    ),
                },
            )
        except LLMError as exc:
            self._evenement("erreur", {"message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            # L'onglet a été fermé : inutile de continuer à générer pour personne. La
            # prochaine écriture lèverait de toute façon.
            pass
        finally:
            # On ne présume pas du type : `stream_with_usage` annonce un itérateur, mais
            # en pratique c'est un générateur — donc fermable, ce qui libère la connexion
            # HTTP du modèle.
            if flux is not None and hasattr(flux, "close"):
                flux.close()
            cls.verrou_tour.release()


class ServeurWeb:
    """Le service web lui-même : une page, un flux, et le TTS hérité."""

    def __init__(
        self,
        tts,
        session: "ChatSession",
        port: int = PORT_WEB,
        charger_transcription: Any = None,
    ) -> None:
        # `staticmethod` : sans lui, la fonction posée dans la classe serait prise pour une
        # méthode et recevrait `self` en premier argument au moment de l'appel.
        #
        # `transcriber`, `verrou_chargement` et `transcriptions` sont reposés ici, et pas
        # hérités : hérités, ils seraient **communs à tous les serveurs du même
        # processus** — un second serveur verrait le Whisper du premier, et son compteur.
        handler = type(
            "_HandlerWebLie",
            (_HandlerWeb,),
            {
                "tts": tts,
                "session": session,
                "charger_transcription": staticmethod(charger_transcription),
                "transcriber": None,
                "verrou_chargement": threading.Lock(),
                "transcriptions": 0,
            },
        )
        self.tts = tts
        self.session = session
        self.charger_transcription = charger_transcription
        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
        # Port réellement pris : avec 0, l'OS en choisit un libre.
        self.port = int(self._httpd.server_address[1])
        self._fil: threading.Thread | None = None

    def demarrer(self) -> str:
        self._fil = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._fil.start()
        return self.url

    @property
    def url(self) -> str:
        return f"http://{distant._ip_locale()}:{self.port}"

    @property
    def url_local(self) -> str:
        """Adresse de boucle : sert aux tests et au dépannage sur la même machine."""
        return f"http://127.0.0.1:{self.port}"

    def arreter(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._fil is not None:
            self._fil.join(timeout=3)
