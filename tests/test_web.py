"""Tests de l'interface web (v1.2).

Aucun réseau, aucun GPU : le flux du modèle est remplacé par une réponse connue, et le
synthétiseur par une doublure. Ce qui est éprouvé ici, c'est le **transport** — ce qui
part réellement sur le fil — pas la qualité du modèle.

Trois choses méritent d'être vérifiées sérieusement :

* les phrases annoncées à la voix se recollent exactement pour former le texte affiché.
  Si les deux divergeaient, l'écran et le son ne diraient pas la même chose — c'est
  précisément pour ça que la découpe est partagée avec la console (`tour.derouler`) ;
* une erreur pendant un tour arrive **dans le flux**, pas en code HTTP : un `EventSource`
  de navigateur ne lit le corps que si la réponse est en 200, donc un 4xx se traduirait
  par « connexion perdue » sans explication ;
* un second tour pendant qu'un premier est en cours est refusé proprement.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

import pytest

import voicechat.web as web
from test_distant import FauxTTS
from test_stt_distant import FauxTranscriber
from voicechat.cli import ChatSession
from voicechat.config import Config
from voicechat.llm import LLMError, Usage

REPONSE = "Bonjour. Voici deux phrases. Et une troisième sans point final"


def faux_stream(base_url=None, model=None, messages=None, **kwargs):
    """Tient lieu de `stream_with_usage` : rend une réponse connue, mot à mot.

    Les morceaux partent **sans** espace final superflu : c'est ce que fait un vrai
    modèle, et le test de recollage des phrases compare les mots un à un.
    """
    usage = Usage()
    usage.prompt_tokens = 120
    usage.completion_tokens = 18
    usage.chars = len(REPONSE)
    mots = REPONSE.split(" ")

    def gen():
        for index, mot in enumerate(mots):
            yield mot + (" " if index < len(mots) - 1 else "")

    return gen(), usage


def faux_stream_erreur(base_url=None, model=None, messages=None, **kwargs):
    def gen():
        yield "Début"
        raise LLMError("serveur LLM injoignable")

    return gen(), Usage()


class FauxTTSQuiEchoue(FauxTTS):
    def synth(self, text: str):  # noqa: ARG002 — nom de la méthode parente
        raise RuntimeError("CUDA out of memory")


# ------------------------------------------------------------------ installation


@pytest.fixture
def serveur(monkeypatch):
    monkeypatch.setattr(web, "stream_with_usage", faux_stream)
    session = ChatSession(
        Config(model="m.gguf", tts=False, n_ctx=8_000, system="Tu es concis.")
    )
    srv = web.ServeurWeb(FauxTTS(), session, port=0)
    srv.demarrer()
    yield srv, session
    srv.arreter()


def flux(srv, question: str) -> str:
    """L'URL du flux, question encodée — une espace brute fait refuser l'URL par urllib."""
    return f"{srv.url_local}/flux?q={quote(question)}"


def lire_sse(url: str) -> list[tuple[str, dict]]:
    """Lit un flux SSE et rend la liste des (évènement, données)."""
    evenements: list[tuple[str, dict]] = []
    with urllib.request.urlopen(url, timeout=15) as reponse:
        nom = ""
        for ligne_brute in reponse:
            ligne = ligne_brute.decode("utf-8").rstrip("\n")
            if ligne.startswith("event: "):
                nom = ligne[len("event: ") :]
            elif ligne.startswith("data: "):
                evenements.append((nom, json.loads(ligne[len("data: ") :])))
    return evenements


def _poster(url: str, charge: dict | None = None, delai: float = 10) -> tuple[int, bytes]:
    corps = json.dumps(charge or {}).encode("utf-8")
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(requete, timeout=delai) as reponse:
            return reponse.status, reponse.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _poster_octets(url: str, octets: bytes, ctype: str = "audio/webm") -> tuple[int, bytes]:
    """Le geste exact du navigateur : des octets audio bruts, sans enveloppe JSON."""
    requete = urllib.request.Request(
        url, data=octets, headers={"Content-Type": ctype}, method="POST"
    )
    try:
        with urllib.request.urlopen(requete, timeout=20) as reponse:
            return reponse.status, reponse.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _wav_16k(secondes: float = 0.4) -> bytes:
    """Un vrai WAV, pour que le décodage du service ait quelque chose à manger."""
    import io as _io

    import numpy as np
    import soundfile as sf

    signal = (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(16000 * secondes)) / 16000))
    memoire = _io.BytesIO()
    sf.write(memoire, signal.astype("float32"), 16000, format="WAV", subtype="PCM_16")
    return memoire.getvalue()


# ------------------------------------------------------------------ la page


def test_la_page_est_servie(serveur):
    srv, _ = serveur
    with urllib.request.urlopen(f"{srv.url_local}/", timeout=10) as reponse:
        assert reponse.status == 200
        assert reponse.headers["Content-Type"].startswith("text/html")
        page = reponse.read().decode("utf-8")
    assert "<title>voicechat</title>" in page
    assert "EventSource" in page  # le client parle bien en SSE
    # Rien d'externe : la page doit s'ouvrir sur un poste sans accès Internet.
    assert "http://" not in page.replace("http://www.w3.org", "")


def test_la_page_n_a_pas_de_dependance_externe(serveur):
    """Une page qui charge un CDN serait inutilisable sur un tailnet isolé."""
    srv, _ = serveur
    with urllib.request.urlopen(f"{srv.url_local}/", timeout=10) as reponse:
        page = reponse.read().decode("utf-8")
    assert "cdn" not in page.lower()
    assert "<script src=" not in page


def test_etat_annonce_modele_et_contexte(serveur):
    srv, _ = serveur
    with urllib.request.urlopen(f"{srv.url_local}/etat", timeout=10) as reponse:
        etat = json.loads(reponse.read())
    assert etat["modele"] == "m"  # nom_court retire le chemin et l'extension
    assert etat["contexte"]["total"] == 8_000
    assert etat["voix"] == "ff_siwis"


# ------------------------------------------------------------------ le tour


def test_le_flux_raconte_le_tour(serveur):
    srv, session = serveur
    evenements = lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    noms = [nom for nom, _ in evenements]
    assert noms[0] == "debut"
    assert noms[-1] == "fin"
    assert "texte" in noms and "phrase" in noms


def test_le_texte_recu_est_la_reponse(serveur):
    srv, _ = serveur
    textes = "".join(d["t"] for n, d in lire_sse(f"{srv.url_local}/flux?q=Bonjour")
                     if n == "texte")
    assert textes == REPONSE


def test_les_phrases_dites_recolent_le_texte(serveur):
    """Le contrôle qui compte : écran et voix doivent dire la même chose."""
    srv, _ = serveur
    evenements = lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    texte = "".join(d["t"] for n, d in evenements if n == "texte")
    phrases = [d["t"] for n, d in evenements if n == "phrase"]
    # On recolle les phrases en normalisant les espaces : le découpage peut déplacer une
    # espace d'un fragment à l'autre, jamais un mot.
    assert " ".join(phrases).split() == texte.split()
    # Le dernier fragment n'avait pas de ponctuation finale : il doit être dit quand même.
    assert phrases[-1] == "Et une troisième sans point final"


def test_la_conversation_est_mémorisée_côté_serveur(serveur):
    srv, session = serveur
    lire_sse(flux(srv, "Première question"))
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant"]
    assert session.messages[1]["content"] == "Première question"
    assert session.messages[2]["content"] == REPONSE


def test_l_ancrage_sur_les_chiffres_du_serveur_est_mis_a_jour(serveur):
    srv, session = serveur
    assert session._ancrage_jetons == 0
    lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    assert session._ancrage_jetons == 120  # ce que le serveur a rendu


def test_le_dernier_morceau_sans_ponctuation_part_quand_même(serveur, monkeypatch):
    monkeypatch.setattr(
        web,
        "stream_with_usage",
        lambda **k: (iter(["Une phrase sans ponctuation finale"]), Usage()),
    )
    srv, _ = serveur
    phrases = [d["t"] for n, d in lire_sse(f"{srv.url_local}/flux?q=Dis") if n == "phrase"]
    assert phrases == ["Une phrase sans ponctuation finale"]


# ------------------------------------------------------------------ pannes


def test_une_question_vide_est_refusée_dans_le_flux(serveur):
    srv, _ = serveur
    evenements = lire_sse(f"{srv.url_local}/flux?q=")
    assert evenements[-1][0] == "erreur"
    assert "vide" in evenements[-1][1]["message"]


def test_une_erreur_du_modele_arrive_dans_le_flux(serveur, monkeypatch):
    """Un 4xx ne dirait rien : EventSource ne lit pas le corps hors 200."""
    monkeypatch.setattr(web, "stream_with_usage", faux_stream_erreur)
    srv, _ = serveur
    evenements = lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    noms = [nom for nom, _ in evenements]
    assert "erreur" in noms
    assert "injoignable" in dict(evenements)["erreur"]["message"]


def test_un_second_tour_simultané_est_refusé(serveur):
    srv, _ = serveur
    assert web._HandlerWeb.verrou_tour.acquire(blocking=True, timeout=1)
    try:
        evenements = lire_sse(f"{srv.url_local}/flux?q=Bonjour")
        assert evenements[-1][0] == "erreur"
        assert "déjà en cours" in evenements[-1][1]["message"]
    finally:
        web._HandlerWeb.verrou_tour.release()


def test_le_verrou_est_relaché_après_une_erreur(serveur, monkeypatch):
    """Sinon la page resterait bloquée jusqu'au redémarrage du serveur."""
    monkeypatch.setattr(web, "stream_with_usage", faux_stream_erreur)
    srv, _ = serveur
    lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    assert not web._HandlerWeb.verrou_tour.locked()
    # Et un tour normal passe derrière.
    monkeypatch.setattr(web, "stream_with_usage", faux_stream)
    assert [n for n, _ in lire_sse(f"{srv.url_local}/flux?q=Encore")][-1] == "fin"


def test_l_audio_ne_doit_pas_attendre_la_fin_du_tour(serveur, monkeypatch):
    """Régression : l'audio démarrait après la réponse complète, au lieu de la suivre.

    Cause : la route `/parle` prenait `verrou`, nom qui est aussi celui du verrou GPU
    hérité du serveur TTS — et que `/flux` garde pendant **tout** le tour. La synthèse de
    la première phrase attendait donc la dernière.

    Le test reproduit le geste du navigateur : il retient la génération en plein milieu,
    puis demande l'audio. La réponse doit venir tout de suite, pas à la fin du tour.
    """
    phrase_prete = threading.Event()
    suite = threading.Event()

    def flux_retenu(base_url=None, model=None, messages=None, **kwargs):
        def gen():
            yield "Bonjour. "  # phrase complète : le navigateur va demander son audio
            phrase_prete.set()
            suite.wait(timeout=20)  # la génération reste en cours, volontairement
            yield "Et voilà la suite."
        usage = Usage()
        usage.prompt_tokens = 12
        return gen(), usage

    monkeypatch.setattr(web, "stream_with_usage", flux_retenu)
    srv, _ = serveur

    def lire() -> None:
        with urllib.request.urlopen(flux(srv, "Bonjour"), timeout=30) as reponse:
            for _ in reponse:
                pass

    fil = threading.Thread(target=lire, daemon=True)
    fil.start()
    assert phrase_prete.wait(timeout=15), "la première phrase n'est jamais arrivée"

    # Le tour est toujours en cours. C'est ici que le bug se voyait.
    debut = time.monotonic()
    try:
        code, corps = _poster(f"{srv.url_local}/parle", {"texte": "Bonjour."}, delai=5)
    finally:
        suite.set()
    attente = time.monotonic() - debut

    assert code == 200
    assert corps[:4] == b"RIFF"
    assert attente < 3.0, (
        f"l'audio a attendu {attente:.1f} s : il est resté derrière le tour en cours"
    )
    fil.join(timeout=20)


# ------------------------------------------------------------------ autres routes


def test_reset_vide_l_historique(serveur):
    srv, session = serveur
    lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    assert len(session.messages) == 3
    code, _ = _poster(f"{srv.url_local}/reset")
    assert code == 200
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "system"
    assert session._ancrage_jetons == 0


def test_reset_remet_les_avertissements_a_zero(serveur):
    srv, session = serveur
    session._oubli_signale = True
    session._contexte_signale = True
    _poster(f"{srv.url_local}/reset")
    assert session._oubli_signale is False
    assert session._contexte_signale is False


def test_le_serveur_tts_reste_disponible(serveur):
    """`/parle` est hérité du serveur TTS : la page doit continuer d'avoir son audio."""
    srv, _ = serveur
    with urllib.request.urlopen(f"{srv.url_local}/sante", timeout=10) as reponse:
        assert json.loads(reponse.read())["pret"] is True
    code, corps = _poster(f"{srv.url_local}/parle", {"texte": "Bonjour."})
    assert code == 200
    assert corps[:4] == b"RIFF"  # en-tête WAV


def test_une_synthese_qui_echoue_remonte_en_500(serveur):
    srv = web.ServeurWeb(
        FauxTTSQuiEchoue(),
        ChatSession(Config(model="m.gguf", tts=False)),
        port=0,
    )
    srv.demarrer()
    try:
        code, corps = _poster(f"{srv.url_local}/parle", {"texte": "Bonjour."})
        assert code == 500
        assert b"CUDA out of memory" in corps
    finally:
        srv.arreter()


def test_route_inconnue_renvoie_404(serveur):
    srv, _ = serveur
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{srv.url_local}/nexistepas", timeout=10)
    assert err.value.code == 404


def test_le_port_zero_prend_un_port_libre():
    a = web.ServeurWeb(FauxTTS(), ChatSession(Config(model="m", tts=False)), port=0)
    b = web.ServeurWeb(FauxTTS(), ChatSession(Config(model="m", tts=False)), port=0)
    try:
        assert a.port != b.port
    finally:
        a._httpd.server_close()
        b._httpd.server_close()


# ---------------------------------------------------------------- le micro (v1.2)
# Le serveur web doit pouvoir transcrire : c'est ce qui permet au micro d'un téléphone de
# servir alors que le téléphone n'a ni GPU ni Whisper. Ces tests passent par les **vraies**
# routes HTTP, avec des octets audio réels — pas par les méthodes internes.


def _serveur_micro(chargeur, monkeypatch, appels: list | None = None):
    """Un serveur web avec un chargeur de transcription, sans jamais charger Whisper."""
    monkeypatch.setattr(web, "stream_with_usage", faux_stream)
    session = ChatSession(Config(model="m.gguf", tts=False))

    def compter():
        if appels is not None:
            appels.append(1)
        return FauxTranscriber(texte="bonjour le monde")

    srv = web.ServeurWeb(
        FauxTTS(), session, port=0,
        charger_transcription=compter if chargeur else None,
    )
    srv.demarrer()
    return srv


def test_le_serveur_web_transcrit(monkeypatch):
    """Le geste du navigateur, de bout en bout : des octets entrent, du texte sort."""
    srv = _serveur_micro(chargeur=True, monkeypatch=monkeypatch)
    try:
        code, corps = _poster_octets(f"{srv.url_local}/transcris", _wav_16k())
        assert code == 200
        infos = json.loads(corps)
        assert infos["texte"] == "bonjour le monde"
        assert infos["langue"] == "fr"
        assert infos["rtf"] == pytest.approx(0.4 / 0.4, abs=0.01)
    finally:
        srv.arreter()


def test_whisper_arrive_a_la_premiere_demande_et_une_seule_fois(monkeypatch):
    """Charger Whisper coûte de la VRAM : une fois, au premier besoin, pas au démarrage."""
    appels: list = []
    srv = _serveur_micro(chargeur=True, monkeypatch=monkeypatch, appels=appels)
    try:
        assert appels == []  # rien n'a été chargé au démarrage

        code, _ = _poster_octets(f"{srv.url_local}/transcris", _wav_16k())
        assert code == 200
        assert len(appels) == 1

        code, _ = _poster_octets(f"{srv.url_local}/transcris", _wav_16k())
        assert code == 200
        assert len(appels) == 1, "le second envoi ne doit pas recharger Whisper"
    finally:
        srv.arreter()


def test_transcription_indisponible_sans_chargeur(monkeypatch):
    """Sans chargeur, la route existe mais le dit clairement au lieu de planter."""
    srv = _serveur_micro(chargeur=False, monkeypatch=monkeypatch)
    try:
        code, corps = _poster_octets(f"{srv.url_local}/transcris", _wav_16k())
        assert code == 503
        assert b"transcription" in corps
    finally:
        srv.arreter()


def test_transcris_corps_vide(monkeypatch):
    srv = _serveur_micro(chargeur=True, monkeypatch=monkeypatch)
    try:
        code, corps = _poster_octets(f"{srv.url_local}/transcris", b"")
        assert code == 400
        assert b"vide" in corps
    finally:
        srv.arreter()


def test_transcris_contenu_illisible(monkeypatch):
    srv = _serveur_micro(chargeur=True, monkeypatch=monkeypatch)
    try:
        code, corps = _poster_octets(
            f"{srv.url_local}/transcris", b"ceci n'est pas de l'audio"
        )
        assert code == 400
        assert b"illisible" in corps
    finally:
        srv.arreter()


def test_l_etat_annonce_si_le_micro_est_possible(monkeypatch):
    """La page a besoin de le savoir **avant** d'enregistrer, pour montrer le bouton."""
    srv = _serveur_micro(chargeur=True, monkeypatch=monkeypatch)
    try:
        with urllib.request.urlopen(f"{srv.url_local}/etat", timeout=10) as reponse:
            infos = json.loads(reponse.read())
        assert infos["transcription"]["disponible"] is True
        assert infos["transcription"]["pret"] is False  # pas encore chargé
        assert infos["transcription"]["modele"] == "?"
    finally:
        srv.arreter()

    srv = _serveur_micro(chargeur=False, monkeypatch=monkeypatch)
    try:
        with urllib.request.urlopen(f"{srv.url_local}/etat", timeout=10) as reponse:
            infos = json.loads(reponse.read())
        assert infos["transcription"]["disponible"] is False
    finally:
        srv.arreter()


def test_deux_serveurs_ne_partagent_pas_le_transcripteur(monkeypatch):
    """Le Whisper d'un serveur ne doit pas apparaître dans l'autre.

    C'est le piège de la classe fabriquée : un attribut hérité de `_HandlerWeb` serait
    commun à tous les serveurs du processus.
    """
    monkeypatch.setattr(web, "stream_with_usage", faux_stream)
    a = web.ServeurWeb(
        FauxTTS(), ChatSession(Config(model="m", tts=False)), port=0,
        charger_transcription=lambda: FauxTranscriber(texte="serveur A"),
    )
    b = web.ServeurWeb(
        FauxTTS(), ChatSession(Config(model="m", tts=False)), port=0,
        charger_transcription=lambda: FauxTranscriber(texte="serveur B"),
    )
    a.demarrer()
    b.demarrer()
    try:
        assert json.loads(_poster_octets(f"{a.url_local}/transcris", _wav_16k())[1])[
            "texte"
        ] == "serveur A"
        assert json.loads(_poster_octets(f"{b.url_local}/transcris", _wav_16k())[1])[
            "texte"
        ] == "serveur B"
    finally:
        a.arreter()
        b.arreter()


def test_port_zero_du_serveur_web_va_jusqu_a_l_os(monkeypatch):
    """`--port 0` doit demander un port libre, pas retomber sur celui par défaut.

    `port or PORT_WEB` avalait le zéro — 0 est faux en Python — et le serveur tombait
    silencieusement sur 8091 : le lancement échouait alors avec « Address already in use »
    si ce port était déjà pris, sans que rien n'explique pourquoi. C'est arrivé pour de
    vrai pendant une vérification.
    """
    from voicechat import cli

    ports: list[int] = []

    class FauxServeur:
        def __init__(self, tts, session, port, charger_transcription=None):
            ports.append(port)

        def demarrer(self) -> str:
            return "http://127.0.0.1:0"

        @property
        def url_local(self) -> str:
            return "http://127.0.0.1:0"

        @property
        def port(self) -> int:
            return 0

        def arreter(self) -> None:
            pass

    class FausseSession:
        def __init__(self, cfg):
            self.cfg = cfg
            self.tts = object()

        def resolve_model(self) -> bool:
            return True

        def setup_voice(self, avec_pipeline: bool = True) -> None:
            pass

    def interrompre(_secondes):
        raise KeyboardInterrupt

    monkeypatch.setattr(web, "ServeurWeb", FauxServeur)
    monkeypatch.setattr(cli, "ChatSession", FausseSession)
    monkeypatch.setattr(cli.time, "sleep", interrompre)

    assert cli.do_web(Config(model="m.gguf", tts=False), port=0) == 0
    assert ports == [0], "le zéro doit arriver tel quel : c'est l'OS qui choisit le port"


def test_la_page_offre_le_micro_et_parle_aux_bonnes_routes():
    """La page et le serveur doivent s'entendre sur les noms de routes.

    Un renommage de route oublié côté page ne se voit qu'en cliquant sur le bouton, à la
    main, dans un navigateur — ce test le voit tout de suite.
    """
    page = (web.PAGE).read_text(encoding="utf-8")
    assert 'id="micro"' in page
    assert 'fetch("transcris"' in page
    assert 'fetch("etat")' in page
    assert "getUserMedia" in page
    # Le cas HTTPS doit être expliqué, pas silencieux : sans contexte sécurisé,
    # `getUserMedia` n'existe même pas et il n'y a aucun message d'erreur à lire.
    assert "HTTPS" in page
