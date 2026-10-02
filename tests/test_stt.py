"""Tests de la couche de transcription (faster-whisper).

Aucun téléchargement de modèle ici : on vérifie la logique autour du modèle
(choix du type de calcul, assemblage des segments, statistiques), pas la qualité
de la reconnaissance — cela se mesure avec `tests/verif_ecoute.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicechat.stt import COMBINAISONS, MODELES, Resultat, Transcriber


# ------------------------------------------------------------ choix du calcul
def test_float16_nest_jamais_propose():
    """Régression : float16 est le choix « évident » et c'est le seul qui échoue ici.

    CTranslate2 refuse float16 sur Pascal :
    « Requested float16 compute type, but the target device or backend do not
    support efficient float16 computation. »
    """
    tous = [compute for essais in COMBINAISONS.values() for _, compute in essais]
    assert "float16" not in tous
    assert "int8_float32" in tous, "c'est le meilleur compromis mesuré (RTF 0,107)"


def test_cuda_retombe_sur_le_cpu_en_dernier_recours():
    for profil in ("auto", "cuda"):
        essais = COMBINAISONS[profil]
        assert essais[-1][0] == "cpu", f"profil {profil} : pas de repli CPU"


def test_profil_cpu_ne_propose_que_du_cpu():
    assert all(device == "cpu" for device, _ in COMBINAISONS["cpu"])


def test_modeles_connus():
    assert "small" in MODELES
    assert "large-v3" in MODELES


# ----------------------------------------------------------------- statistiques
def test_resume_du_resultat():
    resultat = Resultat(texte="bonjour", langue="fr", probabilite_langue=0.98,
                        duree_audio_s=5.0, duree_s=0.6)
    assert resultat.rtf == pytest.approx(0.12)
    resume = resultat.resume()
    assert "5.00 s" in resume and "0.60 s" in resume
    assert "fr" in resume and "98%" in resume


def test_resume_sans_audio_ne_divise_pas_par_zero():
    resultat = Resultat()
    assert resultat.rtf == 0.0
    assert isinstance(resultat.resume(), str)


# ------------------------------------------------------------ faux modèle
class _Info:
    language = "fr"
    language_probability = 0.97


class _Segment:
    def __init__(self, texte: str) -> None:
        self.text = texte


class FauxModele:
    """Tient lieu de WhisperModel : renvoie les segments qu'on lui donne."""

    def __init__(self, morceaux: list[str]) -> None:
        self.morceaux = morceaux
        self.appel: dict | None = None

    def transcribe(self, audio, **kwargs):
        self.appel = {"taille": len(audio), **kwargs}
        return iter([_Segment(m) for m in self.morceaux]), _Info()


def test_assemblage_des_segments():
    transcriber = Transcriber(modele="small", langue="fr")
    transcriber._modele = FauxModele([" Bonjour, ", "ceci est un test. ", "Fin."])
    resultat = transcriber.transcrire(np.zeros(16000, dtype=np.float32))
    assert resultat.texte == "Bonjour, ceci est un test. Fin."
    assert resultat.duree_audio_s == pytest.approx(1.0)
    assert resultat.langue == "fr"
    assert resultat.probabilite_langue == pytest.approx(0.97)


def test_les_parametres_de_transcription_sont_transmis():
    transcriber = Transcriber(modele="small", langue="fr", beam_size=3)
    faux = FauxModele(["ok"])
    transcriber._modele = faux
    transcriber.transcrire(np.zeros(16000, dtype=np.float32))
    assert faux.appel is not None
    assert faux.appel["language"] == "fr"
    assert faux.appel["beam_size"] == 3
    assert faux.appel["vad_filter"] is True, "le VAD interne rogne les silences résiduels"


def test_langue_vide_demande_la_detection_automatique():
    transcriber = Transcriber(modele="small", langue="")
    faux = FauxModele(["hello"])
    transcriber._modele = faux
    transcriber.transcrire(np.zeros(16000, dtype=np.float32))
    assert faux.appel is not None and faux.appel["language"] is None


def test_audio_vide_rend_un_resultat_vide():
    transcriber = Transcriber()
    transcriber._modele = FauxModele(["ne devrait pas être appelé"])
    resultat = transcriber.transcrire(np.zeros(0, dtype=np.float32))
    assert resultat.texte == "" and resultat.duree_s == 0.0


def test_transcrire_sans_modele_leve_une_erreur_claire():
    transcriber = Transcriber()
    assert not transcriber.pret
    with pytest.raises(RuntimeError):
        transcriber.transcrire(np.zeros(16000, dtype=np.float32))


def test_chargement_impossible_donne_un_message_lisible(monkeypatch):
    """Un modèle inexistant ne doit pas lever : on veut un message, pas une trace."""
    transcriber = Transcriber(modele="modele-qui-nexiste-pas-0000", device="cpu")
    assert transcriber.charger() is False
    assert transcriber.erreur
    assert "aucune configuration" in transcriber.erreur


# ------------------------------------------------------------ fichier -> tableau
def test_transcrire_fichier_reechantillonne(tmp_path):
    """Un WAV 24 kHz doit être ramené à 16 kHz avant d'aller au modèle."""
    soundfile = pytest.importorskip("soundfile")
    chemin = tmp_path / "test.wav"
    soundfile.write(str(chemin), np.zeros(24000, dtype=np.float32), 24000)

    transcriber = Transcriber(modele="small", langue="fr")
    faux = FauxModele(["ok"])
    transcriber._modele = faux
    transcriber.transcrire_fichier(str(chemin))
    assert faux.appel is not None
    assert faux.appel["taille"] == 16000, "1 s à 24 kHz doit devenir 1 s à 16 kHz"


def test_transcrire_fichier_stereo_est_converti_en_mono(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    chemin = tmp_path / "stereo.wav"
    soundfile.write(str(chemin), np.zeros((16000, 2), dtype=np.float32), 16000)

    transcriber = Transcriber(modele="small")
    faux = FauxModele(["ok"])
    transcriber._modele = faux
    transcriber.transcrire_fichier(str(chemin))
    assert faux.appel is not None and faux.appel["taille"] == 16000
