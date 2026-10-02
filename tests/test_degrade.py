"""Tests du mode dégradé : quand la synthèse échoue, le texte ne doit pas être perdu.

Le GPU peut lâcher en pleine session (mémoire pleine, pilote qui tombe). Jusqu'à la
v0.6, la phrase concernée était comptée puis **jetée** : le son manquait, et rien ne
le disait. On vérifie ici qu'elle est gardée, signalée, et rejouable.

Ni GPU ni carte son : le synthétiseur et le haut-parleur sont des doublures.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicechat.tts import SpeechPipeline


class FauxTTS:
    """Synthétiseur qui échoue les N premiers appels, puis réussit."""

    def __init__(self, echecs: int = 0) -> None:
        self.echecs_restants = echecs
        self.appels: list[str] = []

    def synth(self, texte: str) -> np.ndarray:
        self.appels.append(texte)
        if self.echecs_restants > 0:
            self.echecs_restants -= 1
            raise RuntimeError("CUDA error: out of memory")
        return np.zeros(2400, dtype=np.float32)  # 0,1 s de silence


class FauxSpeaker:
    """Haut-parleur factice : note ce qui a été joué."""

    def __init__(self) -> None:
        self.joues: list[np.ndarray] = []
        self.flushes = 0

    def play(self, audio: np.ndarray) -> None:
        self.joues.append(audio)

    def wait(self) -> None:
        return None

    def flush(self) -> None:
        self.flushes += 1


@pytest.fixture()
def pipeline():
    """Un pipeline prêt à l'emploi, arrêté en fin de test."""
    tts = FauxTTS()
    speaker = FauxSpeaker()
    p = SpeechPipeline(tts, speaker, cleaner=lambda t: t.strip())
    p.start()
    try:
        yield p, tts, speaker
    finally:
        p.close()


def test_phrase_normale_est_jouee(pipeline):
    p, _, speaker = pipeline
    p.say("Bonjour.")
    p.wait()
    assert len(speaker.joues) == 1


def test_echec_de_synthese_garde_la_phrase(pipeline):
    """Le point de la v0.6 : la phrase est mise de côté, pas jetée."""
    p, tts, speaker = pipeline
    tts.echecs_restants = 1
    p.say("Cette phrase va échouer.")
    p.wait()

    assert speaker.joues == [], "rien ne doit être joué"
    assert p.en_attente == 1
    assert p.en_panne
    assert p.synth_errors == 1


def test_rejouer_remet_en_file(pipeline):
    p, tts, speaker = pipeline
    tts.echecs_restants = 1
    p.say("À réentendre plus tard.")
    p.wait()
    assert p.en_attente == 1

    # Le GPU est revenu : le rejeu doit synthétiser et jouer pour de bon.
    combien = p.rejouer()
    p.wait()

    assert combien == 1
    assert p.en_attente == 0
    assert not p.en_panne
    assert len(speaker.joues) == 1, "la phrase doit être réellement jouée au rejeu"


def test_rejouer_sans_panne_ne_fait_rien(pipeline):
    p, _, _ = pipeline
    assert p.rejouer() == 0


def test_plusieurs_echecs_puis_rejeu_global(pipeline):
    p, tts, speaker = pipeline
    tts.echecs_restants = 3
    for phrase in ("Une.", "Deux.", "Trois."):
        p.say(phrase)
    p.wait()

    assert p.en_attente == 3
    assert p.rejouer() == 3
    p.wait()
    assert p.en_attente == 0
    assert len(speaker.joues) == 3


def test_attente_bornee_garde_les_plus_recentes(pipeline):
    """Une longue session ne doit pas remplir la mémoire : on garde la fin."""
    p, tts, _ = pipeline
    p.max_attente = 3
    tts.echecs_restants = 10
    for i in range(6):
        p.say(f"Phrase {i}.")
    p.wait()
    assert p.en_attente == 3

    # On regarde ce qui est effectivement rejoué plutôt que l'état interne.
    tts.echecs_restants = 0
    tts.appels.clear()
    p.rejouer()
    p.wait()
    assert tts.appels == ["Phrase 3.", "Phrase 4.", "Phrase 5."]


def test_le_vidage_ne_perd_pas_les_phrases_en_attente(pipeline):
    """Ctrl+C vide la file de lecture, mais les phrases gardées restent rejouables.

    Sinon il suffirait d'interrompre une réponse pour perdre définitivement les
    passages que le GPU n'avait pas pu dire.
    """
    p, tts, _ = pipeline
    tts.echecs_restants = 1
    p.say("Perdue par le GPU.")
    p.wait()
    assert p.en_attente == 1

    p.flush()
    assert p.en_attente == 1, "le vidage ne doit pas effacer ce qui attend un rejeu"
    assert p.rejouer() == 1


def test_voix_coupee_n_est_pas_mise_en_attente(pipeline):
    """Une phrase abandonnée à cause d'un Ctrl+C n'est pas une panne : pas de rejeu."""
    p, tts, _ = pipeline
    tts.echecs_restants = 0
    p.say("Une phrase.")
    p.flush()  # vidage avant même la synthèse
    p.wait()
    assert p.en_attente == 0
