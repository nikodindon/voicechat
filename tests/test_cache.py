"""Tests de la réserve d'audio déjà synthétisé (v1.0).

Aucun GPU : un faux pipeline tient lieu de Kokoro. On vérifie surtout deux choses
qu'un cache audio rate facilement :

* la **clé** doit dépendre de la voix, de la vitesse et de la langue — sinon changer
  de voix ressert l'audio de l'ancienne ;
* une entrée **corrompue** ne doit pas casser l'application : on la jette et on
  resynthétise.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicechat.cache import CacheAudio
from voicechat.tts import KokoroTTS


@pytest.fixture()
def cache(tmp_path):
    return CacheAudio(tmp_path / "reserve")


class FauxPipeline:
    """Tient lieu de KPipeline : audio déterministe, appels comptés."""

    def __init__(self) -> None:
        self.appels = 0

    def __call__(self, texte, voice=None, speed=1.0):
        self.appels += 1
        audio = np.full(2400, 0.5, dtype=np.float32)
        return iter([("", "", audio)])


def tts_branche(tmp_path):
    """Un KokoroTTS dont la synthèse est simulée, avec sa réserve."""
    reserve = CacheAudio(tmp_path / "reserve")
    tts = KokoroTTS(voice="ff_siwis", speed=1.0, cache=reserve)
    faux = FauxPipeline()
    tts._pipeline = faux  # type: ignore[assignment]  # doublure de test
    tts._voix_chargee = "ff_siwis"
    tts._loaded_voice = "ff_siwis"
    return tts, faux, reserve


# ------------------------------------------------------------------ la clé
def test_cle_stable():
    assert CacheAudio.cle("Bonjour.", "ff_siwis", 1.0, "f") == CacheAudio.cle(
        "Bonjour.", "ff_siwis", 1.0, "f"
    )


def test_cle_sensible_a_la_voix_et_au_reste():
    """Le piège du cache audio : oublier la voix dans la clé."""
    base = CacheAudio.cle("Bonjour.", "ff_siwis", 1.0, "f")
    assert base != CacheAudio.cle("Bonjour.", "af_heart", 1.0, "f")
    assert base != CacheAudio.cle("Bonjour.", "ff_siwis", 1.2, "f")
    assert base != CacheAudio.cle("Bonjour.", "ff_siwis", 1.0, "e")
    assert base != CacheAudio.cle("Bonsoir.", "ff_siwis", 1.0, "f")


def test_cle_distingue_un_melange_de_la_voix_seule():
    assert CacheAudio.cle("Bonjour.", "ff_siwis:3+ef_dora:1", 1.0, "f") != CacheAudio.cle(
        "Bonjour.", "ff_siwis", 1.0, "f"
    )


# ---------------------------------------------------------- écriture/lecture
def test_ecrire_puis_lire_rend_le_meme_audio(cache):
    audio = np.sin(np.linspace(0, 20, 2400)).astype(np.float32)
    cle = CacheAudio.cle("Bonjour.", "ff_siwis", 1.0, "f")
    assert cache.ecrire(cle, audio) is True

    relu = cache.lire(cle)
    assert relu is not None
    assert relu.shape == audio.shape
    np.testing.assert_array_equal(relu, audio), "le WAV float32 doit être exact"


def test_lecture_absente_ne_plante_pas(cache):
    assert cache.lire("inconnue") is None
    assert cache.misses == 1 and cache.hits == 0


def test_audio_vide_nest_pas_range(cache):
    cle = CacheAudio.cle("vide", "ff_siwis", 1.0, "f")
    assert cache.ecrire(cle, np.zeros(0, dtype=np.float32)) is False
    assert cache.lire(cle) is None


def test_entree_corrompue_est_jettee_et_non_servie(cache):
    """Un WAV tronqué ne doit pas faire échouer l'application."""
    cle = CacheAudio.cle("Bonjour.", "ff_siwis", 1.0, "f")
    cache.dossier.mkdir(parents=True, exist_ok=True)
    cache.chemin(cle).write_bytes(b"ceci n'est pas un wav")

    assert cache.lire(cle) is None
    assert not cache.chemin(cle).exists(), "l'entrée illisible doit être supprimée"


def test_reserve_desactivee_ne_lit_ni_n_ecrit(tmp_path):
    reserve = CacheAudio(tmp_path / "reserve", actif=False)
    cle = CacheAudio.cle("Bonjour.", "ff_siwis", 1.0, "f")
    assert reserve.ecrire(cle, np.ones(100, dtype=np.float32)) is False
    assert reserve.lire(cle) is None
    assert reserve.resume() == "réserve désactivée"


# ------------------------------------------------------------------ élagage
def test_elagage_borne_le_nombre_de_fichiers(tmp_path):
    reserve = CacheAudio(tmp_path / "reserve", max_fichiers=3)
    for i in range(6):
        reserve.ecrire(CacheAudio.cle(f"Phrase {i}.", "v", 1.0, "f"),
                       np.full(100, i, dtype=np.float32))
    assert len(list(reserve.dossier.glob("*.wav"))) == 3


def test_elagage_garde_les_plus_recemment_utilisees(tmp_path):
    """Une phrase relue souvent ne doit pas être évincée par des phrases récentes.

    L'ordre compte : on vieillit A, on écrit B (plus récente), on **relit A** (ce qui
    rafraîchit sa date), puis on écrit C. Au-delà de la limite, l'éviction doit
    partir par B — la plus ancienne au moment du tri — et non par A.
    """
    import os
    import time

    reserve = CacheAudio(tmp_path / "reserve", max_fichiers=2)
    a = CacheAudio.cle("Souvent utilisée.", "v", 1.0, "f")
    b = CacheAudio.cle("Utilisée une fois.", "v", 1.0, "f")
    c = CacheAudio.cle("Toute neuve.", "v", 1.0, "f")

    reserve.ecrire(a, np.ones(100, dtype=np.float32))
    vieux = time.time() - 10_000
    os.utime(reserve.chemin(a), (vieux, vieux))
    reserve.ecrire(b, np.ones(100, dtype=np.float32))

    assert reserve.lire(a) is not None, "relecture de A : sa date est rafraîchie"

    reserve.ecrire(c, np.ones(100, dtype=np.float32))

    assert reserve.chemin(a).exists(), "A vient d'être relue, elle ne doit pas partir"
    assert reserve.chemin(c).exists()
    assert not reserve.chemin(b).exists(), "c'est B, la plus ancienne, qui doit partir"


def test_vider(tmp_path):
    reserve = CacheAudio(tmp_path / "reserve")
    for i in range(3):
        reserve.ecrire(CacheAudio.cle(f"P{i}", "v", 1.0, "f"), np.ones(50, dtype=np.float32))
    assert reserve.vider() == 3
    assert list(reserve.dossier.glob("*.wav")) == []


# --------------------------------------------------------- branchement réel
def test_synthese_deux_fois_ne_synthetise_qu_une(tmp_path):
    """Le vrai test : la deuxième synthèse doit venir de la réserve."""
    tts, faux, reserve = tts_branche(tmp_path)
    premier = tts.synth("Bonjour, ceci est un test.")
    deuxieme = tts.synth("Bonjour, ceci est un test.")

    assert faux.appels == 1, "le pipeline ne doit être appelé qu'une fois"
    assert reserve.hits == 1 and reserve.misses == 1
    np.testing.assert_array_equal(premier, deuxieme)


def test_changer_de_voix_ne_ressert_pas_l_ancien_audio(tmp_path):
    """Régression : c'est LE bug classique d'un cache audio mal clé."""
    tts, faux, reserve = tts_branche(tmp_path)
    tts.synth("Bonjour.")
    tts.voice = "af_heart"
    tts._voix_chargee = "af_heart"
    tts.synth("Bonjour.")

    assert faux.appels == 2, "changer de voix doit resynthétiser"
    assert reserve.hits == 0


def test_changer_de_vitesse_ne_ressert_pas_l_ancien_audio(tmp_path):
    tts, faux, _ = tts_branche(tmp_path)
    tts.synth("Bonjour.")
    tts.speed = 1.5
    tts.synth("Bonjour.")
    assert faux.appels == 2


def test_sans_reserve_tout_est_synthetise(tmp_path):
    tts = KokoroTTS(voice="ff_siwis", cache=None)
    faux = FauxPipeline()
    tts._pipeline = faux  # type: ignore[assignment]  # doublure de test
    tts._voix_chargee = "ff_siwis"
    tts._loaded_voice = "ff_siwis"
    tts.synth("Bonjour.")
    tts.synth("Bonjour.")
    assert faux.appels == 2


def test_temps_economise_est_une_estimation(tmp_path):
    """On ne peut pas mesurer le temps qu'une synthèse n'a pas pris : c'est estimé."""
    tts, _, reserve = tts_branche(tmp_path)
    tts.synth("Bonjour.")
    tts.synth("Bonjour.")
    # 0,1 s d'audio resservie, à un RTF de synthèse mesuré dans la session.
    assert tts.temps_economise_s > 0
    assert reserve.servis_s == pytest.approx(0.1, abs=1e-6)
