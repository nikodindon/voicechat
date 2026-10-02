"""Tests de la spécification de voix : un nom, ou un mélange pondéré.

Aucun chargement de modèle ici : on teste l'analyse de la syntaxe et sa
normalisation. La qualité sonore des mélanges est mesurée par
`tests/bench_voix.py`, qui a besoin du GPU et de Whisper.
"""

from __future__ import annotations

import pytest

from voicechat.tts import analyser_voix, decrire_voix, est_melange


def test_voix_seule():
    assert analyser_voix("ff_siwis") == [("ff_siwis", 1.0)]
    assert not est_melange("ff_siwis")


def test_voix_seule_non_melangee_garde_le_rendu_dorigine():
    """Une voix seule doit passer par le chemin normal de Kokoro, pas par un mélange."""
    parts = analyser_voix("af_heart")
    assert len(parts) == 1
    assert parts[0][1] == 1.0


def test_virgule_donne_un_melange_egal():
    """La syntaxe native de Kokoro (`a,b`) doit donner exactement 50/50."""
    assert analyser_voix("ff_siwis,ef_dora") == [("ff_siwis", 0.5), ("ef_dora", 0.5)]


def test_plus_donne_un_melange_egal():
    assert analyser_voix("ff_siwis+ef_dora") == [("ff_siwis", 0.5), ("ef_dora", 0.5)]


def test_poids_ponderes_normalises():
    parts = analyser_voix("ff_siwis:3+ef_dora:1")
    assert parts[0][0] == "ff_siwis" and parts[0][1] == pytest.approx(0.75)
    assert parts[1][0] == "ef_dora" and parts[1][1] == pytest.approx(0.25)
    assert sum(p for _, p in parts) == pytest.approx(1.0)


def test_poids_quelconques_somment_toujours_a_un():
    """Peu importe l'échelle choisie : c'est le rapport qui compte."""
    attendu = analyser_voix("a:30+b:10")
    assert attendu[0][1] == pytest.approx(0.75)
    assert analyser_voix("a:0.75+b:0.25") == pytest.approx(attendu)


def test_trois_voix():
    parts = analyser_voix("ff_siwis:3+ef_dora:1+if_sara:1")
    assert [p for _, p in parts] == pytest.approx([0.6, 0.2, 0.2])


def test_espaces_et_casse_du_poids():
    assert analyser_voix(" ff_siwis : 3 + ef_dora : 1 ") == analyser_voix(
        "ff_siwis:3+ef_dora:1"
    )


def test_poids_entier_sans_decimal():
    assert analyser_voix("a:2+b:2") == [("a", 0.5), ("b", 0.5)]


def test_est_melange():
    assert est_melange("ff_siwis,ef_dora")
    assert est_melange("ff_siwis:4+ef_dora:1")
    assert est_melange("a,b,c")
    assert not est_melange("a:2")


# ------------------------------------------------------------------- erreurs
def test_specification_vide_refusee():
    with pytest.raises(ValueError, match="aucune voix"):
        analyser_voix("   ")


def test_poids_illisible_refuse():
    with pytest.raises(ValueError, match="poids illisible"):
        analyser_voix("ff_siwis:beaucoup")


def test_poids_nul_ou_negatif_refuse():
    with pytest.raises(ValueError, match="nul ou négatif"):
        analyser_voix("ff_siwis:0")
    with pytest.raises(ValueError, match="nul ou négatif"):
        analyser_voix("ff_siwis:-1")


def test_nom_vide_refuse():
    with pytest.raises(ValueError, match="nom de voix vide"):
        analyser_voix(":3+ef_dora:1")


# ----------------------------------------------------------------- affichage
def test_decrire_voix_seule():
    assert decrire_voix("ff_siwis") == "ff_siwis"


def test_decrire_voix_melangee():
    assert decrire_voix("ff_siwis:3+ef_dora:1") == "ff_siwis 75% + ef_dora 25%"


def test_decrire_voix_illisible_ne_leve_pas():
    """Afficher une erreur ne doit pas provoquer une seconde erreur."""
    assert decrire_voix("ff_siwis:beaucoup") == "ff_siwis:beaucoup"
