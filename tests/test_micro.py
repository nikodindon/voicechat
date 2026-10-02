"""Tests de l'écoute micro : VAD Silero et découpage en énoncés.

Le VAD est testé sur le **vrai** modèle (livré avec faster-whisper) : c'est le seul
moyen de vérifier que le protocole v6 — entrée de 576 échantillons, états ``h``/``c``
séparés — est correctement implémenté. Le découpage en énoncés, lui, est testé avec
un micro et un VAD factices : pas besoin de matériel, et le test reste déterministe.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicechat.micro import FREQUENCE, DetecteurParole, Ecouteur, Microphone

BLOC = DetecteurParole.TAILLE_BLOC
BLOC_S = BLOC / FREQUENCE  # 32 ms


# ------------------------------------------------------------------- VAD réel
@pytest.fixture(scope="module")
def vad():
    detecteur = DetecteurParole()
    if not detecteur.pret:
        pytest.skip(detecteur.erreur)
    return detecteur


def test_protocole_v6_est_bien_celui_attendu():
    """L'entrée fait 576 échantillons (64 de contexte + 512) : sinon le modèle refuse."""
    assert DetecteurParole.TAILLE_BLOC == 512
    assert DetecteurParole.CONTEXTE == 64


def test_vad_sur_du_silence(vad):
    vad.reinitialiser()
    probas = [vad.probabilite(np.zeros(BLOC, dtype=np.float32)) for _ in range(10)]
    assert max(probas) < 0.3, f"le silence ne doit pas être pris pour de la parole : {probas}"


def test_vad_sur_du_bruit_fort(vad):
    """Un bruit de fond élevé ne doit pas déclencher : c'est tout l'intérêt de Silero."""
    rng = np.random.default_rng(1234)
    vad.reinitialiser()
    probas = [
        vad.probabilite((rng.standard_normal(BLOC) * 0.3).astype(np.float32))
        for _ in range(10)
    ]
    assert max(probas) < 0.5, f"bruit blanc pris pour de la parole : {max(probas)}"


def test_vad_reinitialiser_remet_letat_a_zero(vad):
    """Sans remise à zéro, l'état LSTM d'une écoute précédente pollue la suivante."""
    vad.reinitialiser()
    premier = vad.probabilite(np.zeros(BLOC, dtype=np.float32))
    for _ in range(20):
        vad.probabilite((np.random.default_rng(0).standard_normal(BLOC) * 0.2).astype(np.float32))
    vad.reinitialiser()
    apres = vad.probabilite(np.zeros(BLOC, dtype=np.float32))
    assert abs(premier - apres) < 0.01


# ------------------------------------------------------------------ re-chunking
def test_microphone_recoupe_en_blocs_fixes():
    micro = Microphone(taille_bloc=512, prechauffe=0.0)
    micro._file.put(np.arange(2048, dtype=np.float32))
    premier = micro.lire(0.1)
    deuxieme = micro.lire(0.1)
    assert premier.size == 512 and premier[0] == 0
    assert deuxieme.size == 512 and deuxieme[0] == 512


def test_microphone_retourne_none_quand_il_manque_des_echantillons():
    micro = Microphone(taille_bloc=512, prechauffe=0.0)
    micro._file.put(np.zeros(100, dtype=np.float32))  # moins d'un bloc
    assert micro.lire(0.05) is None


def test_microphone_jette_les_blocs_dechauffement():
    """Les premiers blocs après ouverture sont souvent un artefact : on les jette."""
    micro = Microphone(taille_bloc=512)
    micro._a_jeter = 2  # simulateur : deux blocs à jeter
    micro._file.put(np.arange(2048, dtype=np.float32))
    bloc = micro.lire(0.1)
    assert bloc[0] == 1024, "les deux premiers blocs (0..1024) devaient être jetés"


def test_pre_roll_non_entier_ne_perd_pas_de_bloc():
    """Régression : le pré-roll faisait toujours un bloc de moins que demandé.

    Le bloc qui déclenche était ajouté à la deque *avant* le test, donc il évincait
    le plus ancien : on perdait exactement le bloc de pré-roll qu'on voulait garder.
    """
    micro = FauxMicro([bloc_marque(float(i)) for i in range(10)])
    ecouteur_test = Ecouteur(
        micro=micro,
        vad=FauxVad([False, False, False] + [True] * 4 + [False] * 3),
        silence_fin=3 * BLOC_S,
        duree_min=BLOC_S,
        pre_roll=3 * BLOC_S,  # soit exactement 3 blocs
    )
    audio = ecouteur_test.ecouter()
    assert audio is not None
    assert audio[0] == 0.0 and audio[BLOC] == 1.0 and audio[2 * BLOC] == 2.0, (
        "les 3 blocs de pré-roll doivent être conservés, pas 2"
    )


def test_microphone_fermer_vide_la_file():
    micro = Microphone(taille_bloc=512, prechauffe=0.0)
    micro._file.put(np.ones(4096, dtype=np.float32))
    micro.fermer()
    assert micro.lire(0.05) is None


# ------------------------------------------------------- découpage en énoncés
class FauxMicro:
    """Débite des blocs préparés à l'avance."""

    def __init__(self, blocs: list[np.ndarray]) -> None:
        self.blocs = list(blocs)
        self.pret = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def lire(self, delai: float = 0.5):
        return self.blocs.pop(0) if self.blocs else None


class FauxVad:
    """Verdicts préparés à l'avance, un par bloc."""

    TAILLE_BLOC = BLOC

    def __init__(self, verdicts: list[bool]) -> None:
        self.verdicts = list(verdicts)
        self.reinitialisations = 0

    def reinitialiser(self) -> None:
        self.reinitialisations += 1

    def contient_parole(self, bloc: np.ndarray) -> bool:
        return self.verdicts.pop(0) if self.verdicts else False


def bloc_marque(valeur: float) -> np.ndarray:
    """Un bloc dont la première valeur identifie sa position dans le scénario."""
    bloc = np.zeros(BLOC, dtype=np.float32)
    bloc[0] = valeur
    return bloc


def ecouteur(verdicts: list[bool], n_blocs: int, **options) -> tuple[Ecouteur, FauxVad]:
    micro = FauxMicro([bloc_marque(float(i)) for i in range(n_blocs)])
    vad = FauxVad(verdicts)
    reglages = {"silence_fin": 3 * BLOC_S, "duree_min": 2 * BLOC_S, "pre_roll": 2 * BLOC_S}
    reglages.update(options)
    return Ecouteur(micro=micro, vad=vad, **reglages), vad


def test_enonce_simple_est_decoupe():
    # 2 silences, 4 paroles, 3 silences -> fin d'énoncé au 3e silence
    verdicts = [False, False] + [True] * 4 + [False] * 3
    ecouteur_test, vad = ecouteur(verdicts, n_blocs=len(verdicts))
    audio = ecouteur_test.ecouter()
    assert audio is not None
    # 2 blocs de pré-roll + 4 paroles (dont celle qui a déclenché) + 3 silences
    assert audio.size == 9 * BLOC
    assert vad.reinitialisations == 1, "l'état du VAD doit être remis à zéro par écoute"


def test_pre_roll_conserve_le_debut_de_la_phrase():
    """Sans pré-roll on coupe la première syllabe : on vérifie qu'il est bien inclus."""
    verdicts = [False, False, False] + [True] * 4 + [False] * 3
    ecouteur_test, _ = ecouteur(verdicts, n_blocs=len(verdicts), pre_roll=3 * BLOC_S)
    audio = ecouteur_test.ecouter()
    assert audio is not None
    assert audio[0] == 0.0, "le premier bloc du pré-roll (marqueur 0) doit être présent"
    assert audio[BLOC] == 1.0
    assert audio[2 * BLOC] == 2.0


def test_enonce_trop_court_est_ignore():
    verdicts = [True, False, False, False]
    ecouteur_test, _ = ecouteur(verdicts, n_blocs=len(verdicts), duree_min=1.0)
    assert ecouteur_test.ecouter() is None, "un toussotement ne doit pas être transcrit"


def test_aucune_parole_rend_none():
    verdicts = [False] * 5
    ecouteur_test, _ = ecouteur(verdicts, n_blocs=len(verdicts))
    assert ecouteur_test.ecouter(delai_depart=0.0) is None


def test_enonce_tronque_a_la_duree_maximale():
    verdicts = [False] + [True] * 200
    ecouteur_test, _ = ecouteur(verdicts, n_blocs=len(verdicts), duree_max=5 * BLOC_S)
    audio = ecouteur_test.ecouter()
    assert audio is not None
    assert audio.size <= 6 * BLOC, "la capture doit être bornée pour ne pas grandir sans fin"


def test_micro_qui_se_tait_en_pleine_phrase_termine_letude():
    """Si le flux s'arrête (micro débranché), on rend ce qu'on a plutôt que de bloquer."""
    micro = FauxMicro([bloc_marque(0.0), bloc_marque(1.0)])
    vad = FauxVad([True, True])
    ecouteur_test = Ecouteur(
        micro=micro, vad=vad, silence_fin=1.0, duree_min=BLOC_S, pre_roll=BLOC_S
    )
    audio = ecouteur_test.ecouter()
    assert audio is not None and audio.size == 2 * BLOC


def test_on_parole_est_appele_une_fois():
    appels: list[int] = []
    verdicts = [False, True, True, False, False, False]
    micro = FauxMicro([bloc_marque(float(i)) for i in range(len(verdicts))])
    ecouteur_test = Ecouteur(
        micro=micro,
        vad=FauxVad(verdicts),
        silence_fin=3 * BLOC_S,
        duree_min=BLOC_S,
        pre_roll=0.0,
        on_parole=lambda: appels.append(1),
    )
    ecouteur_test.ecouter()
    assert appels == [1], "le marqueur de début de parole ne doit sortir qu'une fois"
