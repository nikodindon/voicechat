"""Tests de la recherche dans les conversations sauvegardées (/cherche, v1.1).

Le point sensible n'est pas de trouver une chaîne : c'est de la trouver **malgré les
accents**. Chercher « resume » sans tomber sur « résumé » serait un échec silencieux,
particulièrement pénible en français.
"""

from __future__ import annotations

import json

import pytest

from voicechat import store


@pytest.fixture(autouse=True)
def dossier_isole(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICECHAT_DATA", str(tmp_path / "donnees"))


def sauver(nom: str, *paires: tuple[str, str], maj: float | None = None) -> None:
    """Enregistre une conversation à partir de paires (rôle, contenu)."""
    messages = [{"role": "system", "content": "Tu es concis."}]
    messages += [{"role": r, "content": c} for r, c in paires]
    store.enregistrer(nom, messages, modele="m.gguf", systeme="Tu es concis.")
    if maj is not None:
        chemin = store.dossier() / f"{store.nom_fichier(nom)}.json"
        data = json.loads(chemin.read_text(encoding="utf-8"))
        data["maj"] = maj
        chemin.write_text(json.dumps(data), encoding="utf-8")


# ------------------------------------------------------------------ recherche


def test_trouve_un_mot_simple():
    sauver("a", ("user", "Comment on configure le serveur ?"))
    trouvailles = store.chercher("serveur")
    assert len(trouvailles) == 1
    assert trouvailles[0].conversation == "a"
    assert "serveur" in trouvailles[0].extrait


def test_insensible_a_la_casse():
    sauver("a", ("user", "Le Serveur répond"))
    assert store.chercher("SERVEUR")
    assert store.chercher("serveur")


def test_insensible_aux_accents_dans_les_deux_sens():
    """« resume » doit trouver « résumé », et « résumé » trouver « resume »."""
    sauver("a", ("user", "Fais-moi un résumé du document"))
    assert store.chercher("resume"), "motif sans accent → texte accentué"
    assert store.chercher("résumé"), "motif accentué → texte accentué"

    sauver("b", ("user", "Le resume de la reunion"))
    assert store.chercher("résumé"), "motif accentué → texte sans accent"


def test_extrait_montre_le_contexte_et_les_troncatures():
    sauver("a", ("user", "x" * 200 + " aiguille " + "y" * 200))
    extrait = store.chercher("aiguille")[0].extrait
    assert "aiguille" in extrait
    assert extrait.startswith("… ") and extrait.endswith(" …")


def test_extrait_sans_troncature_quand_le_texte_est_court():
    sauver("a", ("user", "court avec aiguille dedans"))
    extrait = store.chercher("aiguille")[0].extrait
    assert extrait == "court avec aiguille dedans"


def test_aucun_resultat():
    sauver("a", ("user", "Bonjour"))
    assert store.chercher("introuvable") == []


def test_motif_vide_ne_renvoie_rien_et_ne_plante_pas():
    sauver("a", ("user", "Bonjour"))
    assert store.chercher("") == []
    assert store.chercher("   ") == []


def test_trouve_dans_plusieurs_conversations():
    sauver("a", ("user", "on parle de tailscale"))
    sauver("b", ("assistant", "tailscale est un VPN"))
    trouvailles = store.chercher("tailscale")
    assert {t.conversation for t in trouvailles} == {"a", "b"}


def test_la_limite_est_respectee():
    sauver("a", *[("user", f"mot repete {i}") for i in range(10)])
    assert len(store.chercher("repete", limite=3)) == 3


def test_roles_lisibles():
    sauver("a", ("user", "mot"), ("assistant", "mot aussi"))
    roles = {t.role_lisible() for t in store.chercher("mot")}
    assert roles == {"vous", "ia"}


def test_plusieurs_occurrences_dans_la_meme_conversation():
    sauver("a", ("user", "mot"), ("assistant", "encore mot"))
    assert len(store.chercher("mot")) == 2


def test_l_indice_de_message_est_correct():
    """L'index doit désigner le bon message, pas la position dans le texte."""
    sauver("a", ("user", "rien"), ("assistant", "rien"), ("user", "aiguille"))
    trouvaille = store.chercher("aiguille")[0]
    assert trouvaille.message == 3  # 0 = système, 1 et 2 = les deux « rien »
    conversation = store.charger("a")
    assert "aiguille" in conversation.messages[trouvaille.message]["content"]


def test_conversation_abimee_n_empeche_pas_la_recherche():
    sauver("bon", ("user", "aiguille ici"))
    (store.dossier() / "casse.json").write_text("{pas du json", encoding="utf-8")
    trouvailles = store.chercher("aiguille")
    assert len(trouvailles) == 1
    assert trouvailles[0].conversation == "bon"


def test_ligature_ne_fait_pas_sortir_du_texte():
    """« œ » devient « oe » en NFKD : la position ne doit pas dépasser la chaîne."""
    sauver("a", ("user", "le cœur du sujet est l'aiguille"))
    trouvaille = store.chercher("oeur")  # trouve « cœur » via son équivalent sans accent
    assert trouvaille == [] or trouvaille[0].extrait  # pas d'exception
    assert store.chercher("aiguille")  # la vraie recherche fonctionne toujours


def test_sans_accents_preserve_la_longueur_pour_les_lettres_accentuees():
    """C'est ce qui permet de réutiliser la position trouvée pour découper le texte."""
    assert len(store.sans_accents("éàüç")) == len("éàüç")
    assert store.sans_accents("Résumé") == "resume"


def test_recherche_dans_le_prompt_systeme():
    """Le message système est un message comme un autre : il doit être trouvé."""
    sauver("a", ("user", "Bonjour"))
    trouvailles = store.chercher("concis")
    assert any(t.role_lisible() == "système" for t in trouvailles)


def test_les_conversations_recentes_viennent_dabord():
    import time

    maintenant = time.time()
    sauver("ancienne", ("user", "aiguille"), maj=maintenant - 10_000)
    sauver("recente", ("user", "aiguille"), maj=maintenant)
    assert store.chercher("aiguille")[0].conversation == "recente"
