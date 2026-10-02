"""Tests du serveur TTS partagé et de son client (v1.0).

Aucun GPU, aucun Kokoro : un faux synthétiseur côté serveur rend un tableau
reconnaissable. Ce qui est testé ici, c'est le **transport** — le texte part, le son
revient, et les pannes (serveur absent, erreur de synthèse, texte vide) remontent
avec un message exploitable plutôt qu'un traceback.

Le serveur écoute sur le port 0 : l'OS en choisit un libre, donc pas de collision
entre tests ni avec le vrai service.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from voicechat.distant import SAMPLE_RATE, ServeurTTS, TTSDistant
from voicechat.tts import Synthetiseur

# ------------------------------------------------------------------ doublures


class FauxTTS:
    """Tient lieu de KokoroTTS : rend un motif dépendant du texte."""

    def __init__(self, voix: str = "ff_siwis", echec: Exception | None = None) -> None:
        self.voice = voix
        self.lang_code = "f"
        self.speed = 1.0
        self.device = "cpu"
        self.device_reason = "doublure de test"
        self.load_error = None
        self.cache = None
        self.synth_s = 0.0
        self.audio_s = 0.0
        self.echec = echec
        self.textes: list[str] = []
        self.voix_demandees: list[str] = []

    @property
    def ready(self) -> bool:
        return True

    @property
    def rtf(self) -> float:
        return 0.0

    @property
    def temps_economise_s(self) -> float:
        return 0.0

    def load(self) -> bool:
        return True

    def set_voice(self, voice: str) -> None:
        self.voix_demandees.append(voice)
        self.voice = voice

    def set_lang(self, lang_code: str) -> None:
        self.lang_code = lang_code

    def synth(self, text: str) -> np.ndarray:
        if self.echec is not None:
            raise self.echec
        self.textes.append(text)
        # Motif lié au texte : permet de vérifier que c'est bien la bonne phrase
        # qui a été synthétisée, et pas la précédente restée en mémoire.
        n = 240 + len(text) * 10
        return (np.arange(n, dtype=np.float32) / n) * 0.5


@pytest.fixture
def serveur():
    faux = FauxTTS()
    srv = ServeurTTS(faux, port=0)
    srv.demarrer()
    yield srv, faux
    srv.arreter()


def _poster(url: str, charge: dict | bytes):
    corps = charge if isinstance(charge, bytes) else json.dumps(charge).encode("utf-8")
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": "application/json"}, method="POST"
    )
    return urllib.request.urlopen(requete, timeout=10)


# ============================================================ côté serveur


def test_sante_annonce_le_service(serveur):
    srv, faux = serveur
    with urllib.request.urlopen(f"{srv.url_local}/sante", timeout=10) as reponse:
        infos = json.loads(reponse.read())
    assert reponse.status == 200
    assert infos["pret"] is True
    assert infos["service"] == "voicechat-tts/1"
    assert infos["voix"] == "ff_siwis"
    assert infos["device"] == "cpu"


def test_route_inconnue_renvoie_404(serveur):
    srv, _ = serveur
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{srv.url_local}/nimportequoi", timeout=10)
    assert err.value.code == 404
    assert b"/parle" in err.value.read()


def test_parle_renvoie_un_wav_exploitable(serveur):
    srv, faux = serveur
    with _poster(f"{srv.url_local}/parle", {"texte": "Bonjour."}) as reponse:
        assert reponse.status == 200
        assert reponse.headers["Content-Type"] == "audio/wav"
        brut = reponse.read()

    import soundfile as sf

    audio, taux = sf.read(io.BytesIO(brut), dtype="float32")
    assert taux == SAMPLE_RATE
    assert audio.size == 240 + len("Bonjour.") * 10
    assert faux.textes == ["Bonjour."]


def test_parle_transmet_voix_et_vitesse(serveur):
    srv, faux = serveur
    _poster(
        f"{srv.url_local}/parle",
        {"texte": "Salut.", "voix": "if_sara", "vitesse": 1.4, "langue": "f"},
    ).read()
    assert faux.voix_demandees == ["if_sara"]
    assert faux.speed == 1.4
    assert faux.voice == "if_sara"


def test_parle_sans_texte_renvoie_400(serveur):
    srv, faux = serveur
    with pytest.raises(urllib.error.HTTPError) as err:
        _poster(f"{srv.url_local}/parle", {"texte": "   "})
    assert err.value.code == 400
    assert "texte" in err.value.read().decode()
    assert faux.textes == []  # rien n'a été synthétisé


def test_parle_avec_json_illisible_renvoie_400(serveur):
    srv, _ = serveur
    with pytest.raises(urllib.error.HTTPError) as err:
        _poster(f"{srv.url_local}/parle", b"{ceci n'est pas du json")
    assert err.value.code == 400
    assert "JSON" in err.value.read().decode()


def test_parle_erreur_de_synthese_renvoie_500_lisible():
    """Un GPU qui lâche doit se lire dans le message, pas seulement dans les logs."""
    srv = ServeurTTS(FauxTTS(echec=RuntimeError("CUDA out of memory")), port=0)
    srv.demarrer()
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            _poster(f"{srv.url_local}/parle", {"texte": "Bonjour."})
        assert err.value.code == 500
        assert "CUDA out of memory" in err.value.read().decode()
    finally:
        srv.arreter()


def test_sante_renvoie_503_si_non_charge():
    class PasPret(FauxTTS):
        @property
        def ready(self) -> bool:
            return False

    srv = ServeurTTS(PasPret(), port=0)
    srv.demarrer()
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{srv.url_local}/sante", timeout=10)
        assert err.value.code == 503
        assert "non chargé" in err.value.read().decode()
    finally:
        srv.arreter()


# ============================================================= côté client


def test_client_se_charge_sur_un_serveur_vivant(serveur):
    srv, _ = serveur
    client = TTSDistant(url=srv.url_local)
    assert client.load() is True
    assert client.ready is True
    assert "serveur TTS" in client.device_reason
    assert "ff_siwis" in client.device_reason


def test_client_signale_un_serveur_absent():
    """Le message doit dire quoi faire, pas seulement que ça a échoué."""
    client = TTSDistant(url="http://127.0.0.1:1")  # port 1 : rien n'écoute
    assert client.load() is False
    assert client.ready is False
    assert "injoignable" in (client.load_error or "")
    assert "--serveur-tts" in (client.load_error or "")


def test_aller_retour_du_son(serveur):
    """Le son qui revient est celui du serveur, échantillon pour échantillon."""
    srv, faux = serveur
    client = TTSDistant(url=srv.url_local)
    assert client.load()

    audio = client.synth("Bonjour.")
    attendu = faux.synth("Bonjour.")  # le motif attendu, calculé localement
    assert audio.size == attendu.size
    assert float(np.abs(audio - attendu).max()) == 0.0
    assert client.appels == 1
    assert client.audio_s == pytest.approx(audio.size / SAMPLE_RATE)


def test_client_transmet_la_voix_choisie(serveur):
    srv, faux = serveur
    client = TTSDistant(url=srv.url_local, voice="ff_siwis")
    client.load()
    client.set_voice("af_heart")
    client.synth("Test.")
    assert faux.voix_demandees == ["af_heart"]


def test_client_ne_synthetise_pas_sans_chargement():
    client = TTSDistant(url="http://127.0.0.1:1")
    with pytest.raises(RuntimeError) as err:
        client.synth("Bonjour.")
    assert "non joint" in str(err.value)


def test_client_remonte_une_erreur_du_serveur():
    srv = ServeurTTS(FauxTTS(echec=RuntimeError("CUDA out of memory")), port=0)
    srv.demarrer()
    try:
        client = TTSDistant(url=srv.url_local)
        assert client.load()
        with pytest.raises(RuntimeError) as err:
            client.synth("Bonjour.")
        assert "HTTP 500" in str(err.value)
        assert "CUDA out of memory" in str(err.value)
    finally:
        srv.arreter()


def test_client_respecte_le_contrat_de_synthetiseur():
    """Le client doit pouvoir remplacer KokoroTTS sans rien changer ailleurs."""
    assert isinstance(TTSDistant(url="http://x"), Synthetiseur)


def test_le_serveur_accepte_deux_clients_de_suite(serveur):
    """Deux clients : le second ne doit pas trouver un serveur cassé."""
    srv, faux = serveur
    a = TTSDistant(url=srv.url_local)
    b = TTSDistant(url=srv.url_local)
    assert a.load() and b.load()
    a.synth("Phrase A.")
    b.synth("Phrase B.")
    a.synth("Phrase A.")
    assert faux.textes == ["Phrase A.", "Phrase B.", "Phrase A."]


def test_le_port_zero_prend_un_port_libre():
    """Deux services sur la même machine ne doivent pas se marcher dessus."""
    a = ServeurTTS(FauxTTS(), port=0)
    b = ServeurTTS(FauxTTS(), port=0)
    try:
        assert a.port != b.port
        assert a.port > 0
    finally:
        a._httpd.server_close()
        b._httpd.server_close()
