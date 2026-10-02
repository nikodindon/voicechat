"""Tests de la reprise réseau (v0.6) : réessais, bascule, erreurs définitives.

On utilise un faux serveur qui **échoue volontairement** les N premières requêtes,
puis répond normalement : c'est le seul moyen de vérifier une politique de reprise
sans dépendre d'une vraie panne.

    .venv/bin/python -m pytest tests/test_reprise.py -q
"""

from __future__ import annotations

import socket
import sys
import threading
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_llm_server import Handler  # noqa: E402

from voicechat.llm import (  # noqa: E402
    Connexion,
    LLMError,
    _cibles,
    _temporaire,
    premier_serveur,
    stream_with_usage,
)


class ServeurFragile(Handler):
    """Faux serveur qui renvoie ``code_echec`` les ``echecs_restants`` fois suivantes.

    L'interception se fait dans ``do_GET``/``do_POST``, **pas** dans ``_send`` : le
    chemin de streaming du faux serveur écrit directement dans ``wfile`` sans passer
    par ``_send``. Intercepter ``_send`` laissait donc toutes les requêtes de chat
    réussir — le serveur n'échouait jamais, et les tests de reprise ne testaient rien.
    """

    echecs_restants = 0
    code_echec = 503
    requetes = 0

    def _echouer(self) -> bool:
        type(self).requetes += 1
        if type(self).echecs_restants > 0:
            type(self).echecs_restants -= 1
            self._send(
                type(self).code_echec, b'{"error":"pas maintenant"}', "application/json"
            )
            return True
        return False

    def do_GET(self) -> None:  # type: ignore[override]
        if self._echouer():
            return
        super().do_GET()

    def do_POST(self) -> None:  # type: ignore[override]
        if self._echouer():
            return
        super().do_POST()


@pytest.fixture()
def serveur_fragile():
    """Fabrique : ``serveur_fragile(echecs=2, code=503)`` → URL du serveur."""
    serveurs: list[tuple[ThreadingHTTPServer, threading.Thread]] = []

    def _creer(echecs: int = 0, code: int = 503) -> str:
        ServeurFragile.echecs_restants = echecs
        ServeurFragile.code_echec = code
        ServeurFragile.requetes = 0
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), ServeurFragile)
        fil = threading.Thread(target=httpd.serve_forever, daemon=True)
        fil.start()
        serveurs.append((httpd, fil))
        return f"http://127.0.0.1:{httpd.server_address[1]}/v1"

    yield _creer

    for httpd, fil in serveurs:
        httpd.shutdown()
        httpd.server_close()
        fil.join(timeout=3)


@pytest.fixture()
def serveur_sain():
    """Un faux serveur qui répond normalement, pour servir de secours."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fil = threading.Thread(target=httpd.serve_forever, daemon=True)
    fil.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        fil.join(timeout=3)


def url_morte() -> str:
    """Une URL dont le port est fermé : personne n'écoute, connexion refusée."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def messages() -> list[dict]:
    return [{"role": "user", "content": "bonjour"}]


def lire_flux(**kwargs) -> str:
    flux, _ = stream_with_usage(**kwargs)
    return "".join(flux)


# --------------------------------------------------------------- fonctions pures
def test_cibles_normalise_et_dedoublonne():
    assert _cibles("http://a/v1/") == ["http://a/v1"]
    assert _cibles(["http://a/v1", "http://b/v1"]) == ["http://a/v1", "http://b/v1"]
    assert _cibles(["http://a/v1", "http://a/v1/", "http://b/v1"]) == [
        "http://a/v1",
        "http://b/v1",
    ]
    assert _cibles(["", "  ", "http://a/v1"]) == ["http://a/v1"]


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x/v1", code, "msg", {}, None)


def test_erreurs_temporaires_reessayees():
    assert _temporaire(_http(503))
    assert _temporaire(_http(500))
    assert _temporaire(_http(429))
    assert _temporaire(urllib.error.URLError("refusé"))
    assert _temporaire(TimeoutError("trop long"))


def test_erreurs_definitives_non_reessayees():
    assert not _temporaire(_http(404))
    assert not _temporaire(_http(400))
    assert not _temporaire(_http(401))


def test_connexion_bascule_et_resume():
    directe = Connexion(url="http://a/v1")
    assert not directe.bascule
    assert directe.resume() == "http://a/v1"

    basculee = Connexion(url="http://b/v1", echecs=["http://a/v1 : refus"])
    assert basculee.bascule
    assert "b/v1" in basculee.resume() and "1 cible" in basculee.resume()


# --------------------------------------------------------------------- réessais
def test_reessai_sur_erreur_temporaire(serveur_fragile):
    """Deux 503 puis la réponse : la reprise doit passer inaperçue pour l'appelant."""
    url = serveur_fragile(echecs=2, code=503)
    essais: list[int] = []
    texte = lire_flux(
        base_url=url,
        model="m",
        messages=messages(),
        essais=4,
        delai=0.02,
        on_essai=lambda numero, attente, raison: essais.append(numero),
    )
    assert texte, "la réponse doit finir par arriver"
    assert essais == [1, 2], f"deux avis de reprise attendus, obtenus {essais}"
    assert ServeurFragile.requetes == 3, "deux échecs puis une réussite"


def test_avis_de_reprise_recu_a_chaque_attente(serveur_fragile):
    """L'avis sert à éviter l'effet « blocage » : il doit porter le délai et la raison."""
    url = serveur_fragile(echecs=1, code=503)
    vus: list[tuple[int, float, str]] = []
    lire_flux(
        base_url=url,
        model="m",
        messages=messages(),
        essais=3,
        delai=0.02,
        on_essai=lambda n, a, r: vus.append((n, a, r)),
    )
    assert len(vus) == 1
    numero, attente, raison = vus[0]
    assert numero == 1 and attente == pytest.approx(0.02)
    assert "503" in raison


def test_erreur_definitive_n_est_pas_reessayee(serveur_fragile):
    """Un 404 ne changera pas en insistant : une seule requête doit partir."""
    url = serveur_fragile(echecs=5, code=404)
    with pytest.raises(LLMError, match="404"):
        lire_flux(base_url=url, model="m", messages=messages(), essais=3, delai=0.02)
    assert ServeurFragile.requetes == 1, "un 404 ne doit pas être retenté"


def test_epuisement_des_essais_remonte_l_erreur(serveur_fragile):
    url = serveur_fragile(echecs=99, code=503)
    with pytest.raises(LLMError, match="503"):
        lire_flux(base_url=url, model="m", messages=messages(), essais=3, delai=0.02)
    assert ServeurFragile.requetes == 3, "exactement `essais` tentatives"


def test_le_delai_double_a_chaque_essai(serveur_fragile):
    url = serveur_fragile(echecs=99, code=503)
    attentes: list[float] = []
    with pytest.raises(LLMError):
        lire_flux(
            base_url=url,
            model="m",
            messages=messages(),
            essais=4,
            delai=0.01,
            on_essai=lambda n, a, r: attentes.append(a),
        )
    assert attentes == pytest.approx([0.01, 0.02, 0.04])


# --------------------------------------------------------------------- bascule
def test_bascule_vers_le_secours(serveur_sain):
    """Cible principale morte, secours vivant : la réponse doit venir du secours."""
    connexion = Connexion()
    texte = lire_flux(
        base_url=[url_morte(), serveur_sain],
        model="m",
        messages=messages(),
        essais=1,
        connexion=connexion,
    )
    assert texte
    assert connexion.url == serveur_sain
    assert connexion.bascule
    assert len(connexion.echecs) == 1


def test_pas_de_bascule_quand_la_cible_principale_repond(serveur_sain):
    connexion = Connexion()
    lire_flux(
        base_url=[serveur_sain, url_morte()],
        model="m",
        messages=messages(),
        essais=1,
        connexion=connexion,
    )
    assert connexion.url == serveur_sain
    assert not connexion.bascule, "aucun échec : inutile de le signaler"


def test_toutes_les_cibles_muettes(serveur_fragile):
    morte = url_morte()
    with pytest.raises(LLMError):
        lire_flux(
            base_url=[morte, url_morte()],
            model="m",
            messages=messages(),
            essais=1,
        )


def test_le_secours_sert_meme_si_le_principal_repond_mal(serveur_fragile, serveur_sain):
    """Un serveur qui répond 503 en boucle doit être abandonné pour le secours."""
    casse = serveur_fragile(echecs=99, code=503)
    connexion = Connexion()
    texte = lire_flux(
        base_url=[casse, serveur_sain],
        model="m",
        messages=messages(),
        essais=2,
        delai=0.02,
        connexion=connexion,
    )
    assert texte
    assert connexion.url == serveur_sain


def test_premier_serveur_choisit_le_bon(serveur_fragile, serveur_sain):
    modeles, cible = premier_serveur([url_morte(), serveur_sain], timeout=2.0)
    assert modeles == ["fake-local-model"]
    assert cible == serveur_sain


def test_premier_serveur_aucune_cible():
    with pytest.raises(LLMError):
        premier_serveur([url_morte()], timeout=1.0)
