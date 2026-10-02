"""Tests de la couche CLI : cohérence de l'aide et politique de complétion.

Ces tests ne lancent ni serveur ni modèle : on n'instancie que la session, qui ne
touche au réseau que si on le lui demande (et c'est neutralisé ici).
"""

from __future__ import annotations

import pytest

from voicechat import cli
from voicechat.config import Config


@pytest.fixture()
def session(tmp_path, monkeypatch):
    """Une session sans réseau, données et profils isolés dans tmp_path."""
    monkeypatch.setenv("VOICECHAT_DATA", str(tmp_path / "donnees"))
    monkeypatch.setenv("VOICECHAT_PROFILS", str(tmp_path / "profils"))
    return cli.ChatSession(Config())


# ------------------------------------------------------------- aide / commandes
def test_commandes_et_aide_coherentes():
    """L'aide est générée depuis COMMANDES : aucune commande ne peut être oubliée."""
    assert cli.HELP
    for nom, _ in cli.COMMANDES:
        assert nom.split()[0] in cli.HELP


def test_noms_commandes_sans_argument():
    """Les noms utilisés pour la complétion n'incluent pas les arguments, et sont uniques."""
    attendus = {nom.split()[0] for nom, _ in cli.COMMANDES}
    assert set(cli.NOMS_COMMANDES) == attendus, "aucune commande oubliée ni inventée"
    assert cli.NOMS_COMMANDES == sorted(attendus), "ordre stable, pour un Tab prévisible"
    assert len(cli.NOMS_COMMANDES) == len(set(cli.NOMS_COMMANDES)), "pas de doublon"
    assert all(nom.startswith("/") and " " not in nom for nom in cli.NOMS_COMMANDES)


# ------------------------------------------------------------------ complétion
def test_completeur_ignore_le_texte_ordinaire(session):
    """Un message normal n'a rien à compléter."""
    assert session._completeur("bonjour") == []
    assert session._completeur("") == []


def test_completeur_propose_les_commandes(session):
    candidats = session._completeur("/sa")
    assert "/save" in candidats
    assert all(c.startswith("/") for c in candidats)


def test_completeur_renvoie_les_candidats_du_contexte(session, monkeypatch):
    """Le completeur fournit des candidats ; c'est l'éditeur qui filtre par préfixe."""
    monkeypatch.setattr(cli, "list_voices", lambda: ["ff_siwis", "af_heart"])
    assert session._completeur("/voice ") == ["ff_siwis", "af_heart"]
    assert session._completeur("/voice a") == ["ff_siwis", "af_heart"]
    # et rien à compléter hors d'une commande
    assert session._completeur("bonjour") == []


def test_completeur_complete_les_conversations(session):
    from voicechat import store

    store.enregistrer("ma-conv", [{"role": "user", "content": "x"}])
    assert session._completeur("/load ") == ["ma-conv"]
    assert session._completeur("/forget ") == ["ma-conv"]


def test_completeur_complete_les_profils(session):
    from voicechat import store

    store.enregistrer_profil("brain-wash", "Tu réponds court.")
    candidats = session._completeur("/profil ")
    assert "brain-wash" in candidats
    assert "save" in candidats


def test_completeur_tts_et_langue(session):
    assert sorted(session._completeur("/tts ")) == ["off", "on"]
    assert "f" in session._completeur("/lang ")


def test_completeur_voix_hors_ligne_reste_utile(session, monkeypatch):
    """Si Hugging Face est injoignable, on propose au moins la voix courante."""
    monkeypatch.setattr(cli, "list_voices", lambda: [])
    assert session._completeur("/voice ") == [session.cfg.voice]


def test_completeur_ne_plante_pas_si_les_voix_echouent(session, monkeypatch):
    def boum():
        raise RuntimeError("réseau coupé")

    monkeypatch.setattr(cli, "list_voices", boum)
    assert session._completeur("/voice ") == [session.cfg.voice]


# --------------------------------------------------------------- conversation
def test_voix_coupee_sans_pipeline_ne_plante_pas(session):
    """Ctrl+C avant que la voix soit chargée (ou en --no-tts) doit être sans effet."""
    session.speech = None
    session._couper_voix()


def test_appliquer_remplace_l_etat(session):
    from voicechat import store

    conv = store.Conversation(
        nom="essai",
        messages=[
            {"role": "system", "content": "ancien prompt"},
            {"role": "user", "content": "salut"},
            {"role": "assistant", "content": "bonjour"},
        ],
        systeme="prompt du fichier",
        modele="m.gguf",
    )
    session._appliquer(conv)
    assert session.messages[0] == {"role": "system", "content": "prompt du fichier"}
    assert len(session.messages) == 3, "un seul message système en tête"
    assert session.conversation == "essai"
