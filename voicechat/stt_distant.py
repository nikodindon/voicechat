"""Service de transcription partagé, et son client (v1.2).

Symétrique du serveur TTS de la v1.0, mais dans l'autre sens : ici le client envoie de
l'**audio** et reçoit du **texte**. C'est la pièce qui permet à un client léger — un
téléphone, un navigateur — de parler sans embarquer Whisper.

Deux routes, donc :

    GET  /sante        → {"service": "voicechat-stt/1", "pret": true, "modele": "small"}
    POST /transcris    → corps : octets audio bruts (wav, flac, ogg, webm, m4a…)
                         réponse  {"texte": "…", "langue": "fr", "probabilite": 0.99,
                                   "duree_audio_s": 6.1, "duree_s": 0.7, "rtf": 0.11}

Le client (``STTDistant``) a la même forme que ``stt.Transcriber`` : le programme peut
donc transcrire localement ou à distance sans savoir laquelle des deux il utilise.
"""

from __future__ import annotations

import io
import threading
from http.server import ThreadingHTTPServer

import numpy as np

from .service import HandlerService
from .stt import Resultat, lire_audio

PORT_STT = 8092
TITRE_STT = "voicechat-stt/1"


# ============================================================ côté serveur
def decoder_audio(octets: bytes) -> np.ndarray:
    """Octets audio → float32 mono 16 kHz, quel que soit le conteneur.

    Deux chemins, dans cet ordre :

    1. **libsndfile** (``stt.lire_audio``) : exact et rapide pour WAV, FLAC, OGG ;
    2. **PyAV**, via ``faster_whisper.audio.decode_audio`` : c'est ce qui permet d'avaler
       du **webm/opus** ou du m4a — exactement ce que produit le micro d'un navigateur
       (``MediaRecorder``). Sans ce repli, la page devrait fabriquer du WAV elle-même,
       c'est-à-dire refaire à la main ce que le navigateur fait très bien.
    """
    try:
        return lire_audio(io.BytesIO(octets))
    except Exception:
        # Formats exotiques : PyAV sait presque tout lire, au prix d'un détour.
        from faster_whisper.audio import decode_audio

        return np.asarray(
            decode_audio(io.BytesIO(octets), sampling_rate=16000), dtype=np.float32
        )


class RoutesTranscription(HandlerService):
    """Le service de transcription, réutilisable dans n'importe quel serveur.

    Même raison que pour la plomberie HTTP (`service.py`) : le serveur web doit pouvoir
    offrir `/transcris` sans en recopier la logique. Un client léger — un téléphone — n'a
    ainsi qu'une seule adresse à connaître pour parler **et** pour écouter, et il n'existe
    qu'un seul décodage audio à corriger le jour où quelque chose cloche.

    S'utilise comme mixin : ``class MonServeur(AutreChose, RoutesTranscription)``.
    """

    transcriber = None
    # Nommé précisément, et surtout **différent** de tout verrou hérité : la v1.2.1 a
    # montré ce que coûte un nom réutilisé (le verrou GPU écrasé mettait la synthèse de
    # la première phrase derrière la fin du tour).
    verrou_transcription = threading.Lock()
    transcriptions = 0

    # ----------------------------------------------------------------------- état
    def etat_transcription(self) -> dict:
        """Ce que le service peut dire de lui-même, sans rien transcrire."""
        transcriber = type(self).transcriber
        return {
            "service": TITRE_STT,
            "pret": bool(transcriber is not None and transcriber.pret),
            "modele": getattr(transcriber, "modele", "?"),
            "device": getattr(transcriber, "device", "?"),
            "langue": getattr(transcriber, "langue", "") or "auto",
            "transcriptions": type(self).transcriptions,
        }

    # ---------------------------------------------------------------------- route
    def servir_transcris(self) -> None:
        """``POST /transcris`` : des octets audio entrent, du texte sort."""
        transcriber = type(self).transcriber
        if transcriber is None or not transcriber.pret:
            self._erreur(503, "transcripteur non chargé")
            return

        octets = self._lire_corps()
        if not octets:
            self._erreur(400, "corps vide : envoyer les octets audio bruts")
            return

        try:
            audio = decoder_audio(octets)
        except Exception as exc:
            self._erreur(400, f"audio illisible ({exc})")
            return
        if audio.size == 0:
            self._erreur(400, "audio vide après décodage")
            return

        try:
            with type(self).verrou_transcription:
                resultat = transcriber.transcrire(audio)
            type(self).transcriptions += 1
        except Exception as exc:
            self._erreur(500, f"transcription impossible : {exc}")
            return

        self._json(200, _resultat_en_json(resultat))


class _HandlerSTT(RoutesTranscription):
    """Le service STT seul, quand on le lance pour d'autres postes (``--serveur-stt``)."""

    def do_GET(self) -> None:  # noqa: N802 (nom imposé par http.server)
        if self.path.rstrip("/") not in ("/sante", ""):
            self._erreur(404, "routes : GET /sante, POST /transcris")
            return
        etat = self.etat_transcription()
        if not etat["pret"]:
            self._erreur(503, "transcripteur non chargé")
            return
        self._json(200, etat)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/transcris":
            self._erreur(404, "routes : GET /sante, POST /transcris")
            return
        self.servir_transcris()


def _resultat_en_json(resultat: Resultat) -> dict:
    return {
        "texte": resultat.texte,
        "langue": resultat.langue,
        "probabilite": round(resultat.probabilite_langue, 3),
        "duree_audio_s": round(resultat.duree_audio_s, 3),
        "duree_s": round(resultat.duree_s, 3),
        "rtf": round(resultat.rtf, 3),
    }


class ServeurSTT:
    """Le service en lui-même, démarré depuis un transcripteur déjà chargé."""

    def __init__(self, transcriber, port: int = PORT_STT) -> None:
        handler = type("_HandlerSTTLie", (_HandlerSTT,), {"transcriber": transcriber})
        self.transcriber = transcriber
        self._httpd = ThreadingHTTPServer(("0.0.0.0", port), handler)
        self.port = int(self._httpd.server_address[1])
        self._fil: threading.Thread | None = None

    def demarrer(self) -> str:
        self._fil = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._fil.start()
        return self.url

    @property
    def url(self) -> str:
        from .distant import _ip_locale

        return f"http://{_ip_locale()}:{self.port}"

    @property
    def url_local(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def arreter(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._fil is not None:
            self._fil.join(timeout=3)


# ============================================================= côté client
class STTDistant:
    """Transcription déportée : même forme que ``stt.Transcriber``.

    Le programme peut donc appeler l'un ou l'autre sans savoir lequel
    (``pret``, ``charger()``, ``transcrire()``, ``transcrire_fichier()``).
    """

    def __init__(self, url: str, langue: str = "", timeout: float = 180.0) -> None:
        self.url = url.rstrip("/")
        self.langue = langue
        self.timeout = timeout
        self.modele = "distant"
        self.device = "distant"
        self.erreur: str | None = None
        self._pret = False
        self.transcriptions = 0
        self.duree_s = 0.0

    # ------------------------------------------------------------------ chargement
    def charger(self) -> bool:
        """Interroge /sante. Un service muet n'est pas fatal : on le dit et on continue."""
        import json
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.url}/sante", timeout=5.0) as reponse:
                infos = json.loads(reponse.read().decode("utf-8"))
        except Exception as exc:
            self.erreur = (
                f"service STT injoignable ({self.url}) : {exc}\n"
                "  → le lancer sur le poste GPU : voicechat --serveur-stt"
            )
            return False
        if not infos.get("pret"):
            self.erreur = f"service STT pas prêt : {infos}"
            return False
        self.modele = str(infos.get("modele") or "distant")
        self._pret = True
        return True

    @property
    def pret(self) -> bool:
        return self._pret

    # ------------------------------------------------------------------ transcription
    def transcrire(self, audio: np.ndarray) -> Resultat:
        """Envoie un tableau float32 mono 16 kHz, et rend le texte.

        Le tableau est converti en WAV : c'est le format le plus sûr à transporter, et
        libsndfile le relit exactement. Un client qui a déjà des octets compressés
        (webm d'un navigateur) doit utiliser ``transcrire_octets``.
        """
        import soundfile as sf

        tampon = io.BytesIO()
        sf.write(tampon, np.asarray(audio, dtype=np.float32), 16000, subtype="FLOAT",
                 format="WAV")
        return self.transcrire_octets(tampon.getvalue())

    def transcrire_fichier(self, chemin: str) -> Resultat:
        """Lit un fichier et l'envoie tel quel, sans le décoder ici."""
        from pathlib import Path

        return self.transcrire_octets(Path(chemin).read_bytes())

    def transcrire_octets(self, octets: bytes) -> Resultat:
        import json
        import time
        import urllib.error
        import urllib.request

        if not self._pret:
            raise RuntimeError(self.erreur or "service STT non joint")

        requete = urllib.request.Request(
            f"{self.url}/transcris",
            data=octets,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        debut = time.monotonic()
        try:
            with urllib.request.urlopen(requete, timeout=self.timeout) as reponse:
                infos = json.loads(reponse.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise RuntimeError(f"service STT : HTTP {exc.code} {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"service STT injoignable ({self.url}) : {exc}") from exc

        self.transcriptions += 1
        self.duree_s += time.monotonic() - debut
        return Resultat(
            texte=str(infos.get("texte") or ""),
            langue=str(infos.get("langue") or ""),
            probabilite_langue=float(infos.get("probabilite") or 0.0),
            duree_audio_s=float(infos.get("duree_audio_s") or 0.0),
            duree_s=float(infos.get("duree_s") or 0.0),
        )
