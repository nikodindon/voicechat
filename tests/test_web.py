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


def _poster(url: str, charge: dict | None = None) -> tuple[int, bytes]:
    corps = json.dumps(charge or {}).encode("utf-8")
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(requete, timeout=10) as reponse:
            return reponse.status, reponse.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


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
    assert web._HandlerWeb.verrou.acquire(blocking=True, timeout=1)
    try:
        evenements = lire_sse(f"{srv.url_local}/flux?q=Bonjour")
        assert evenements[-1][0] == "erreur"
        assert "déjà en cours" in evenements[-1][1]["message"]
    finally:
        web._HandlerWeb.verrou.release()


def test_le_verrou_est_relaché_après_une_erreur(serveur, monkeypatch):
    """Sinon la page resterait bloquée jusqu'au redémarrage du serveur."""
    monkeypatch.setattr(web, "stream_with_usage", faux_stream_erreur)
    srv, _ = serveur
    lire_sse(f"{srv.url_local}/flux?q=Bonjour")
    assert not web._HandlerWeb.verrou.locked()
    # Et un tour normal passe derrière.
    monkeypatch.setattr(web, "stream_with_usage", faux_stream)
    assert [n for n, _ in lire_sse(f"{srv.url_local}/flux?q=Encore")][-1] == "fin"


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
