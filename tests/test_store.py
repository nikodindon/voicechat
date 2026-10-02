"""Tests de la persistance des conversations (/save, /load, /conversations)."""

from __future__ import annotations

import json

import pytest

from voicechat import store


@pytest.fixture(autouse=True)
def dossier_isole(tmp_path, monkeypatch):
    """Chaque test travaille dans son propre dossier de données."""
    monkeypatch.setenv("VOICECHAT_DATA", str(tmp_path / "donnees"))


MESSAGES = [
    {"role": "system", "content": "Tu es concis."},
    {"role": "user", "content": "Bonjour"},
    {"role": "assistant", "content": "Salut !"},
]


def test_enregistrer_puis_charger():
    chemin = store.enregistrer("test", MESSAGES, modele="m.gguf", systeme="Tu es concis.")
    assert chemin.exists()
    conv = store.charger("test")
    assert conv.messages == MESSAGES
    assert conv.modele == "m.gguf"
    assert conv.echanges == 2  # hors message système


def test_fichier_json_lisible():
    chemin = store.enregistrer("test", MESSAGES)
    data = json.loads(chemin.read_text(encoding="utf-8"))
    assert data["version"] == store.VERSION
    assert data["messages"] == MESSAGES
    assert "cree" in data and "maj" in data


def test_charger_inexistant():
    with pytest.raises(FileNotFoundError, match="aucune conversation"):
        store.charger("jamais-ecrite")


def test_nom_protege_contre_la_traversee_de_chemin():
    """Un nom comme « ../../etc/passwd » ne doit pas sortir du dossier de données."""
    assert "/" not in store.nom_fichier("../../etc/passwd")
    assert store.nom_fichier("../../etc/passwd") == "etc-passwd"

    chemin = store.enregistrer("../../etc/passwd", MESSAGES)
    assert chemin.parent == store.dossier()
    assert store.dossier() in chemin.parents


def test_nom_vide_refuse():
    with pytest.raises(ValueError):
        store.nom_fichier("   ")
    with pytest.raises(ValueError):
        store.nom_fichier("///")


def test_accents_et_espaces_normalises():
    assert store.nom_fichier("Ma conversation d'été") == "Ma-conversation-d-t"


def test_lister_trie_du_plus_recent_au_plus_ancien():
    store.enregistrer("ancienne", MESSAGES)
    # forcage des dates : le tri ne doit pas dépendre de la vitesse du disque
    chemin_vieux = store.dossier() / "ancienne.json"
    data = json.loads(chemin_vieux.read_text(encoding="utf-8"))
    data["maj"] = 1_000_000.0
    chemin_vieux.write_text(json.dumps(data), encoding="utf-8")

    store.enregistrer("recente", MESSAGES)
    noms = [c["nom"] for c in store.lister()]
    assert noms[0] == "recente"
    assert "ancienne" in noms


def test_lister_sans_dossier():
    assert store.lister() == []


def test_derniere_sans_rien():
    assert store.derniere() is None


def test_derniere():
    store.enregistrer("une", MESSAGES)
    assert store.derniere().nom == "une"


def test_supprimer():
    store.enregistrer("jetable", MESSAGES)
    assert store.supprimer("jetable") is True
    assert store.supprimer("jetable") is False
    assert store.derniere() is None


def test_fichier_corrompu_ignore_dans_la_liste():
    chemin = store.enregistrer("valide", MESSAGES)
    (store.dossier() / "casse.json").write_text("{ceci n'est pas du json", encoding="utf-8")

    noms = [c["nom"] for c in store.lister()]
    assert noms == ["valide"], "un fichier illisible ne doit pas casser /conversations"
    assert chemin.exists()


def test_charger_json_valide_mais_mauvais_schema():
    store.dossier().mkdir(parents=True, exist_ok=True)
    (store.dossier() / "etrange.json").write_text('{"rien": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="messages"):
        store.charger("etrange")


def test_messages_invalides_filtres_au_chargement():
    """Des messages au mauvais format sont écartés plutôt que de tout faire planter."""
    store.dossier().mkdir(parents=True, exist_ok=True)
    (store.dossier() / "partiel.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "ok"},
                    {"role": "inconnu", "content": "à jeter"},
                    {"role": "assistant"},  # pas de contenu
                    "pas un dictionnaire",
                ]
            }
        ),
        encoding="utf-8",
    )
    conv = store.charger("partiel")
    assert conv.messages == [{"role": "user", "content": "ok"}]


def test_ecriture_atomique_ne_laisse_pas_de_tmp():
    store.enregistrer("propre", MESSAGES)
    restes = list(store.dossier().glob("*.tmp"))
    assert restes == []


def test_enregistrer_deux_fois_ecrase():
    store.enregistrer("meme", MESSAGES)
    store.enregistrer("meme", MESSAGES[:2])
    assert store.charger("meme").echanges == 1


def test_variable_env_prise_en_compte(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICECHAT_DATA", str(tmp_path / "ailleurs"))
    assert store.dossier() == tmp_path / "ailleurs" / "conversations"
