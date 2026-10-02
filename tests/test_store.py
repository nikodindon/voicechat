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


# --------------------------------------------------------------------- profils
@pytest.fixture()
def profils_isoles(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICECHAT_PROFILS", str(tmp_path / "profils"))


def test_profil_enregistrer_puis_charger(profils_isoles):
    chemin = store.enregistrer_profil("brain-wash", "Tu réponds en français, court.")
    assert chemin.exists()
    assert store.charger_profil("brain-wash") == "Tu réponds en français, court."


def test_profil_liste(profils_isoles):
    store.enregistrer_profil("un", "a")
    store.enregistrer_profil("deux", "b")
    assert store.lister_profils() == ["deux", "un"]


def test_profil_inexistant(profils_isoles):
    with pytest.raises(FileNotFoundError, match="aucun profil"):
        store.charger_profil("jamais-ecrit")


def test_profil_vide_refuse(profils_isoles):
    with pytest.raises(ValueError):
        store.enregistrer_profil("vide", "   ")


def test_profil_txt_accepte(profils_isoles):
    dossier = store.dossier_profils()
    dossier.mkdir(parents=True, exist_ok=True)
    (dossier / "brut.txt").write_text("prompt brut", encoding="utf-8")
    assert store.lister_profils() == ["brut"]
    assert store.charger_profil("brut") == "prompt brut"


def test_profil_nom_assaini(profils_isoles):
    chemin = store.enregistrer_profil("../../etc/passwd", "contenu")
    assert chemin.parent == store.dossier_profils()
    assert store.lister_profils() == ["etc-passwd"]


def test_lister_profils_sans_dossier(profils_isoles):
    assert store.lister_profils() == []


# ---------------------------------------------------------------------- export
def test_export_markdown(tmp_path):
    chemin = store.exporter_markdown(
        MESSAGES, tmp_path / "conv.md", titre="Ma session", modele="m.gguf", voix="ff_siwis"
    )
    texte = chemin.read_text(encoding="utf-8")
    assert texte.startswith("# Ma session")
    assert "**Modèle** : m.gguf" in texte
    assert "**Voix** : ff_siwis" in texte
    assert "## Prompt système" in texte
    assert "Tu es concis." in texte
    assert "**Vous**" in texte and "Bonjour" in texte
    assert "**Assistant**" in texte and "Salut !" in texte


def test_export_cree_les_dossiers(tmp_path):
    chemin = store.exporter_markdown(MESSAGES, tmp_path / "sous" / "dossier" / "conv.md")
    assert chemin.exists()


def test_export_ignore_les_messages_hors_echange(tmp_path):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "résultat technique"},  # rôle non géré
        {"role": "assistant", "content": "   "},  # vide
        {"role": "assistant", "content": "réponse"},
    ]
    texte = store.exporter_markdown(messages, tmp_path / "c.md").read_text(encoding="utf-8")
    assert "résultat technique" not in texte
    assert "question" in texte and "réponse" in texte


def test_export_conversation_vide(tmp_path):
    chemin = store.exporter_markdown([], tmp_path / "vide.md")
    assert chemin.exists()
    assert "## Échanges" in chemin.read_text(encoding="utf-8")


def test_export_sans_voix_n_affiche_pas_la_ligne(tmp_path):
    texte = store.exporter_markdown(MESSAGES, tmp_path / "c.md").read_text(encoding="utf-8")
    assert "**Voix**" not in texte


# ------------------------------------------------- profils : en-tête de réglages
def test_profil_sans_entete_est_un_simple_prompt():
    profil = store.analyser_profil("Tu es un assistant concis.\n")
    assert profil.prompt == "Tu es un assistant concis."
    assert profil.voix is None and profil.vitesse is None and profil.langue is None


def test_profil_avec_entete():
    profil = store.analyser_profil(
        "voix: ff_siwis:3+ef_dora:1\n"
        "vitesse: 1.15\n"
        "langue: f\n"
        "\n"
        "Tu es un assistant concis.\n"
    )
    assert profil.prompt == "Tu es un assistant concis."
    assert profil.voix == "ff_siwis:3+ef_dora:1"
    assert profil.vitesse == 1.15
    assert profil.langue == "f"


def test_profil_entete_sans_ligne_vide():
    """Le premier réglage peut toucher le prompt : on ne le mange pas."""
    profil = store.analyser_profil("voix: af_heart\nTu réponds en anglais.\n")
    assert profil.voix == "af_heart"
    assert profil.prompt == "Tu réponds en anglais."


def test_profil_cle_inconnue_laissee_dans_le_prompt():
    """Une clé non reconnue ne doit jamais être consommée : ce serait perdre du texte."""
    profil = store.analyser_profil("raisonnement: très long\nTu es concis.\n")
    assert profil.voix is None
    assert profil.prompt == "raisonnement: très long\nTu es concis."


def test_profil_vitesse_illisible_ignoree_sans_planter():
    profil = store.analyser_profil("vitesse: rapide\n\nTu es concis.\n")
    assert profil.vitesse is None
    assert profil.prompt == "Tu es concis."


def test_profil_voix_vide_vaut_aucune_voix():
    profil = store.analyser_profil("voix:\n\nTu es concis.\n")
    assert profil.voix is None


def test_profil_aller_retour_avec_voix(profils_isoles):
    store.enregistrer_profil("duo", "Tu es concis.", voix="ff_siwis:3+ef_dora:1",
                            vitesse=1.15)
    profil = store.lire_profil("duo")
    assert profil.prompt == "Tu es concis."
    assert profil.voix == "ff_siwis:3+ef_dora:1"
    assert profil.vitesse == 1.15


def test_profil_sans_reglage_nécrit_pas_dentete(profils_isoles):
    """Un profil qui ne change rien reste un fichier de prompt, comme avant la v0.5."""
    chemin = store.enregistrer_profil("simple", "Tu es concis.")
    assert chemin.read_text(encoding="utf-8") == "Tu es concis.\n"


def test_charger_profil_reste_compatible(profils_isoles):
    """`charger_profil` (utilisé ailleurs) rend le prompt, sans les réglages."""
    store.enregistrer_profil("duo", "Tu es concis.", voix="af_heart")
    assert store.charger_profil("duo") == "Tu es concis."
