"""Serveur TTS partagé et son client (v1.0).

L'idée : le poste léger (portable, pas de GPU) envoie du **texte**, un poste qui a
le GPU renvoie de l'**audio**. Le poste léger n'a donc plus besoin de torch ni de
Kokoro — il ne lui reste que la lecture du son.

Le protocole tient en deux routes, dans les deux sens :

    GET  /sante          → {"pret": true, "voix": "…", "device": "cuda", "rtf": 0.1}
    POST /parle          → corps JSON {"texte", "voix", "vitesse", "langue"}
                           réponse  audio/wav (float32, 24 kHz mono)

Le format des deux côtés est du WAV float32 : exact au bit près après relecture, et
lisible à l'oreille dans n'importe quel lecteur si un doute apparaît.

Côté serveur, la réserve d'audio (``CacheAudio``) fait tout l'intérêt du montage :
une phrase déjà dite n'est pas resynthétisée, même pour un autre client.
"""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SAMPLE_RATE = 24000
PORT_DEFAUT = 8090
TITRE = "voicechat-tts/1"


# ============================================================ côté serveur
class _Handler(BaseHTTPRequestHandler):
    """Routes du service. Le synthétiseur est posé sur le serveur (attribut de classe)."""

    protocol_version = "HTTP/1.1"
    tts = None  # rempli par `servir()`
    verrou = threading.Lock()  # un seul appel au GPU à la fois
    appels = 0

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (nom imposé)
        """Silence : le serveur ne pollue pas le terminal de ses clients."""
        return

    # ------------------------------------------------------------- utilitaires
    def _envoyer(self, code: int, corps: bytes, ctype: str) -> None:
        self.send_response(code)
        # Le charset ne vaut que pour le JSON : l'annoncer sur audio/wav serait faux.
        if ctype.startswith("application/json"):
            ctype = f"{ctype}; charset=utf-8"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def _erreur(self, code: int, message: str) -> None:
        # ensure_ascii=False : sans ça, « synthétiseur non chargé » devient
        # « synth\u00e9tiseur » dans le corps, illisible au curl et dans un navigateur.
        corps = json.dumps({"erreur": message}, ensure_ascii=False).encode("utf-8")
        self._envoyer(code, corps, "application/json")

    # ------------------------------------------------------------------ routes
    def do_GET(self) -> None:  # noqa: N802 (nom imposé par http.server)
        if self.path.rstrip("/") not in ("/sante", ""):
            self._erreur(404, "routes : GET /sante, POST /parle")
            return
        tts = type(self).tts
        if tts is None or not tts.ready:
            self._erreur(503, "synthétiseur non chargé")
            return
        corps = json.dumps(
            {
                "service": TITRE,
                "pret": True,
                "voix": getattr(tts, "voice", "?"),
                "device": getattr(tts, "device", "?"),
                "rtf": round(float(getattr(tts, "rtf", 0.0)), 3),
                "reprises": _reprises(tts),
                "appels": type(self).appels,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self._envoyer(200, corps, "application/json")

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/parle":
            self._erreur(404, "routes : GET /sante, POST /parle")
            return
        tts = type(self).tts
        if tts is None or not tts.ready:
            self._erreur(503, "synthétiseur non chargé")
            return

        taille = int(self.headers.get("Content-Length", 0))
        try:
            demande = json.loads(self.rfile.read(taille) or b"{}")
        except json.JSONDecodeError:
            self._erreur(400, "corps JSON illisible")
            return

        texte = str(demande.get("texte") or "").strip()
        if not texte:
            self._erreur(400, "champ « texte » manquant ou vide")
            return

        voix = demande.get("voix")
        vitesse = demande.get("vitesse")
        langue = demande.get("langue")
        try:
            # Le GPU ne doit jamais servir deux synthèses en même temps : un client
            # qui arrive pendant une phrase attend son tour plutôt que de faire
            # échouer les deux.
            with type(self).verrou:
                if voix and voix != getattr(tts, "voice", None):
                    tts.set_voice(str(voix))
                if langue and langue != getattr(tts, "lang_code", None):
                    tts.set_lang(str(langue))
                if vitesse:
                    tts.speed = float(vitesse)
                audio = tts.synth(texte)
            type(self).appels += 1
        except Exception as exc:
            self._erreur(500, f"synthèse impossible : {exc}")
            return

        if audio is None or len(audio) == 0:
            self._erreur(500, "synthèse vide")
            return

        try:
            import soundfile as sf

            tampon = io.BytesIO()
            sf.write(tampon, np.asarray(audio, dtype=np.float32), SAMPLE_RATE,
                     subtype="FLOAT", format="WAV")
        except Exception as exc:
            self._erreur(500, f"encodage WAV impossible : {exc}")
            return
        self._envoyer(200, tampon.getvalue(), "audio/wav")


def _reprises(tts) -> int:
    cache = getattr(tts, "cache", None)
    return int(getattr(cache, "hits", 0)) if cache is not None else 0


class ServeurTTS:
    """Le service en lui-même, démarré depuis un TTS déjà chargé."""

    def __init__(self, tts, port: int = PORT_DEFAUT) -> None:
        self.tts = tts
        handler = type("_HandlerLie", (_Handler,), {"tts": tts})
        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
        # Le port réellement pris : avec 0, l'OS en choisit un libre (tests, ou deux
        # instances du service sur la même machine).
        self.port = int(self._httpd.server_address[1])
        self._fil: threading.Thread | None = None

    def demarrer(self) -> str:
        self._fil = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._fil.start()
        return self.url

    @property
    def url(self) -> str:
        return f"http://{_ip_locale()}:{self.port}"

    @property
    def url_local(self) -> str:
        """Adresse de boucle : sert aux tests et au dépannage sur la même machine."""
        return f"http://127.0.0.1:{self.port}"

    def arreter(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._fil is not None:
            self._fil.join(timeout=3)


def _ip_locale() -> str:
    """L'adresse à annoncer : celle de la route par défaut, sinon localhost.

    On ouvre un socket UDP vers une adresse publique : ça ne envoie rien, mais ça
    fait choisir à l'OS l'interface qui serait utilisée — donc la bonne adresse à
    communiquer à l'autre poste.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("100.64.0.1", 1))  # n'importe quelle adresse non locale
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


# ============================================================= côté client
class TTSDistant:
    """Synthétiseur vu d'un autre poste : même interface que ``KokoroTTS``.

    Se présente comme un KokoroTTS pour le reste du programme (mêmes attributs, même
    méthode ``synth``), mais chaque phrase part en HTTP et l'audio revient. Le poste
    local n'a donc besoin ni de torch, ni de Kokoro, ni de GPU.
    """

    def __init__(self, url: str, voice: str = "ff_siwis", lang_code: str = "f",
                 speed: float = 1.0, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.voice = voice
        self.lang_code = lang_code
        self.speed = float(speed)
        self.timeout = timeout
        self.device = "distant"
        self.device_reason = f"serveur TTS {self.url}"
        self.load_error: str | None = None
        self.cache = None
        self.synth_s = 0.0
        self.audio_s = 0.0
        self.appels = 0
        self._pret = False

    # ------------------------------------------------------------------ chargement
    def load(self) -> bool:
        """Interroge /sante. Un serveur muet n'est pas fatal : on le dit et on continue."""
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.url}/sante", timeout=5.0) as reponse:
                infos = json.loads(reponse.read().decode("utf-8"))
        except Exception as exc:
            self.load_error = (
                f"serveur TTS injoignable ({self.url}) : {exc}\n"
                "  → le lancer sur le poste GPU : voicechat --serveur-tts"
            )
            return False
        if not infos.get("pret"):
            self.load_error = f"serveur TTS pas prêt : {infos}"
            return False
        self.device_reason = (
            f"serveur TTS {self.url} (voix {infos.get('voix')}, "
            f"device {infos.get('device')})"
        )
        self._pret = True
        return True

    @property
    def ready(self) -> bool:
        return self._pret

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    def set_lang(self, lang_code: str) -> None:
        self.lang_code = lang_code

    # ------------------------------------------------------------------ synthèse
    def synth(self, text: str) -> np.ndarray:
        """Texte → audio, en passant par le réseau."""
        import time
        import urllib.error
        import urllib.request

        if not self._pret:
            raise RuntimeError(self.load_error or "serveur TTS non joint")

        charge = json.dumps(
            {
                "texte": text,
                "voix": self.voice,
                "vitesse": self.speed,
                "langue": self.lang_code,
            }
        ).encode("utf-8")
        requete = urllib.request.Request(
            f"{self.url}/parle", data=charge,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        debut = time.monotonic()
        try:
            with urllib.request.urlopen(requete, timeout=self.timeout) as reponse:
                brut = reponse.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise RuntimeError(f"serveur TTS : HTTP {exc.code} {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"serveur TTS injoignable ({self.url}) : {exc}") from exc

        try:
            import soundfile as sf

            audio, _ = sf.read(io.BytesIO(brut), dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"audio illisible renvoyé par {self.url} : {exc}") from exc

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        self.synth_s += time.monotonic() - debut
        self.audio_s += audio.size / SAMPLE_RATE
        self.appels += 1
        return audio

    @property
    def rtf(self) -> float:
        """Ici, le « temps réel » inclut le réseau — c'est ce que vit l'utilisateur."""
        return self.synth_s / self.audio_s if self.audio_s > 0 else 0.0

    @property
    def temps_economise_s(self) -> float:
        return 0.0  # la réserve vit côté serveur
