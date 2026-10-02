"""Tests du service de transcription distant et de son client (v1.2).

Aucun GPU, aucun Whisper : le transcripteur est une doublure. Ce qui est éprouvé ici,
c'est le **transport** — l'audio part, le texte revient — et la conversion des formats,
qui est la partie où l'on se trompe facilement (stéréo, 44,1 kHz, conteneur exotique).

Le décodage a deux chemins : libsndfile pour les formats classiques, PyAV en repli pour
ceux que produit un navigateur (webm/opus). Ces tests couvrent le premier ; le second est
vérifié dans `verif_stt.py`, avec un vrai fichier.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from voicechat.stt import Resultat, Transcription
from voicechat.stt_distant import ServeurSTT, STTDistant

# ------------------------------------------------------------------ doublures


class FauxTranscriber:
    """Tient lieu de Whisper : rend un texte fixe et retient ce qu'il a reçu."""

    def __init__(self, texte: str = "bonjour le monde", echec: Exception | None = None) -> None:
        self.modele = "small"
        self.device = "cpu"
        self.langue = "fr"
        self.compute_reel = "int8"
        self.erreur = None
        self._pret = True
        self.texte = texte
        self.echec = echec
        self.recus: list[np.ndarray] = []

    @property
    def pret(self) -> bool:
        return self._pret

    def charger(self) -> bool:
        return True

    def transcrire(self, audio: np.ndarray) -> Resultat:
        if self.echec is not None:
            raise self.echec
        self.recus.append(np.asarray(audio, dtype=np.float32))
        return Resultat(
            texte=self.texte,
            langue="fr",
            probabilite_langue=0.98,
            duree_audio_s=float(audio.size) / 16000,
            duree_s=0.4,
        )

    def transcrire_fichier(self, chemin: str) -> Resultat:  # pragma: no cover
        raise AssertionError("le service ne doit pas passer par un fichier")


class PasPret(FauxTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self._pret = False


def wav(secondes: float = 1.0, frequence: int = 16000, canaux: int = 1) -> bytes:
    """Un vrai WAV, à la fréquence et au nombre de canaux demandés."""
    n = int(secondes * frequence)
    signal = (np.sin(np.linspace(0, 60, n)) * 0.3).astype(np.float32)
    donnees = np.stack([signal] * canaux, axis=1) if canaux > 1 else signal
    tampon = io.BytesIO()
    sf.write(tampon, donnees, frequence, subtype="FLOAT", format="WAV")
    return tampon.getvalue()


@pytest.fixture
def serveur():
    faux = FauxTranscriber()
    srv = ServeurSTT(faux, port=0)
    srv.demarrer()
    yield srv, faux
    srv.arreter()


def _poster(url: str, corps: bytes, ctype: str = "application/octet-stream"):
    requete = urllib.request.Request(
        url, data=corps, headers={"Content-Type": ctype}, method="POST"
    )
    try:
        with urllib.request.urlopen(requete, timeout=30) as reponse:
            return reponse.status, reponse.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# ============================================================ côté serveur


def test_sante_annonce_le_service(serveur):
    srv, _ = serveur
    with urllib.request.urlopen(f"{srv.url_local}/sante", timeout=10) as reponse:
        infos = json.loads(reponse.read())
    assert reponse.status == 200
    assert infos["service"] == "voicechat-stt/1"
    assert infos["pret"] is True
    assert infos["modele"] == "small"
    assert infos["langue"] == "fr"


def test_sante_renvoie_503_si_non_charge():
    srv = ServeurSTT(PasPret(), port=0)
    srv.demarrer()
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{srv.url_local}/sante", timeout=10)
        assert err.value.code == 503
        assert "non chargé" in err.value.read().decode()
    finally:
        srv.arreter()


def test_transcris_rend_le_texte(serveur):
    srv, faux = serveur
    code, corps = _poster(f"{srv.url_local}/transcris", wav(1.0))
    infos = json.loads(corps)
    assert code == 200
    assert infos["texte"] == "bonjour le monde"
    assert infos["langue"] == "fr"
    assert infos["probabilite"] == 0.98
    assert infos["duree_audio_s"] == pytest.approx(1.0, abs=0.05)
    assert infos["rtf"] == pytest.approx(0.4, abs=0.01)
    assert len(faux.recus) == 1


def test_le_son_arrive_en_mono_16_khz(serveur):
    """Le service doit convertir : un micro envoie souvent du stéréo 44,1 kHz."""
    srv, faux = serveur
    code, _ = _poster(f"{srv.url_local}/transcris", wav(1.0, frequence=44100, canaux=2))
    assert code == 200
    recu = faux.recus[0]
    assert recu.ndim == 1, "le stéréo doit être ramené en mono"
    assert recu.size == pytest.approx(16000, abs=200), "doit être rééchantillonné à 16 kHz"
    assert recu.dtype == np.float32


def test_une_autre_frequence_est_acceptee(serveur):
    srv, faux = serveur
    code, _ = _poster(f"{srv.url_local}/transcris", wav(0.5, frequence=8000))
    assert code == 200
    assert faux.recus[0].size == pytest.approx(8000, abs=200)


def test_corps_vide_renvoie_400(serveur):
    srv, faux = serveur
    code, corps = _poster(f"{srv.url_local}/transcris", b"")
    assert code == 400
    assert "octets audio" in corps.decode()
    assert faux.recus == []


def test_audio_illisible_renvoie_400(serveur):
    """Du texte envoyé à la place d'un son doit donner un message, pas une trace."""
    srv, faux = serveur
    code, corps = _poster(f"{srv.url_local}/transcris", b"ceci n'est pas de l'audio")
    assert code == 400
    assert "illisible" in corps.decode()
    assert faux.recus == []


def test_erreur_de_transcription_renvoie_500():
    srv = ServeurSTT(FauxTranscriber(echec=RuntimeError("CUDA out of memory")), port=0)
    srv.demarrer()
    try:
        code, corps = _poster(f"{srv.url_local}/transcris", wav(0.2))
        assert code == 500
        assert b"CUDA out of memory" in corps
    finally:
        srv.arreter()


def test_route_inconnue_renvoie_404(serveur):
    srv, _ = serveur
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(f"{srv.url_local}/nexistepas", timeout=10)
    assert err.value.code == 404
    assert b"/transcris" in err.value.read()


def test_le_port_zero_prend_un_port_libre():
    a = ServeurSTT(FauxTranscriber(), port=0)
    b = ServeurSTT(FauxTranscriber(), port=0)
    try:
        assert a.port != b.port
    finally:
        a._httpd.server_close()
        b._httpd.server_close()


# ============================================================= côté client


def test_client_se_charge_sur_un_service_vivant(serveur):
    srv, _ = serveur
    client = STTDistant(url=srv.url_local)
    assert client.charger() is True
    assert client.pret is True
    assert client.modele == "small"


def test_client_signale_un_service_absent():
    """Le message doit dire quoi faire, pas seulement que ça a échoué."""
    client = STTDistant(url="http://127.0.0.1:1")
    assert client.charger() is False
    assert "injoignable" in (client.erreur or "")
    assert "--serveur-stt" in (client.erreur or "")


def test_aller_retour_d_un_tableau(serveur):
    srv, faux = serveur
    client = STTDistant(url=srv.url_local)
    assert client.charger()
    signal = np.zeros(16000, dtype=np.float32)
    resultat = client.transcrire(signal)
    assert resultat.texte == "bonjour le monde"
    assert faux.recus[0].size == pytest.approx(16000, abs=50)


def test_aller_retour_d_un_fichier(serveur, tmp_path):
    srv, faux = serveur
    chemin = tmp_path / "extrait.wav"
    chemin.write_bytes(wav(0.5))
    client = STTDistant(url=srv.url_local)
    client.charger()
    resultat = client.transcrire_fichier(str(chemin))
    assert resultat.texte == "bonjour le monde"
    assert faux.recus[0].size == pytest.approx(8000, abs=100)


def test_client_refuse_de_travailler_sans_chargement():
    client = STTDistant(url="http://127.0.0.1:1")
    with pytest.raises(RuntimeError) as err:
        client.transcrire(np.zeros(1600, dtype=np.float32))
    assert "non joint" in str(err.value)


def test_client_remonte_une_erreur_du_service():
    srv = ServeurSTT(FauxTranscriber(echec=RuntimeError("CUDA out of memory")), port=0)
    srv.demarrer()
    try:
        client = STTDistant(url=srv.url_local)
        assert client.charger()
        with pytest.raises(RuntimeError) as err:
            client.transcrire(np.zeros(1600, dtype=np.float32))
        assert "HTTP 500" in str(err.value)
        assert "CUDA out of memory" in str(err.value)
    finally:
        srv.arreter()


def test_le_client_respecte_le_contrat_de_transcription():
    """Le client doit pouvoir remplacer le Whisper local sans rien changer ailleurs."""
    assert isinstance(STTDistant(url="http://x"), Transcription)


def test_le_transcripteur_local_respecte_le_meme_contrat():
    """Les deux sont interchangeables : c'est tout l'intérêt du contrat."""
    from voicechat.stt import Transcriber

    assert isinstance(Transcriber(modele="small"), Transcription)


def test_le_client_compte_ses_transcriptions(serveur):
    srv, _ = serveur
    client = STTDistant(url=srv.url_local)
    client.charger()
    for _ in range(3):
        client.transcrire(np.zeros(1600, dtype=np.float32))
    assert client.transcriptions == 3
    assert client.duree_s >= 0.0
