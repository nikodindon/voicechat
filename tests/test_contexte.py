"""Tests du comptage de contexte en tokens (v1.2).

Le défaut corrigé : le client comptait les messages **à l'aveugle**, avec une limite de
24 messages. Mesuré sur le serveur du projet, 25 messages ne pèsent que 1 646 tokens —
5 % d'un contexte de 32 768. On jetait donc des messages vingt fois trop tôt, sans
jamais prévenir quand un échange dense approchait vraiment de la limite.

Aucun réseau ici : `ChatSession` se construit hors ligne, et `n_ctx` est fourni à la
main, comme s'il venait de `/props`.
"""

from __future__ import annotations

import pytest

from voicechat.cli import ChatSession, _jetons_estimes
from voicechat.config import Config


def session(**reglages) -> ChatSession:
    """Une session hors ligne, avec un contexte connu d'avance."""
    base = {"model": "m.gguf", "tts": False, "n_ctx": 10_000}
    return ChatSession(Config(**{**base, **reglages}))


def echanges(nb: int, taille: int = 400) -> list[dict]:
    """Un historique factice : `nb` échanges (question + réponse)."""
    messages: list[dict] = [{"role": "system", "content": "Tu es concis."}]
    for i in range(nb):
        messages.append({"role": "user", "content": f"question {i} " + "a" * taille})
        messages.append({"role": "assistant", "content": f"réponse {i} " + "b" * taille})
    return messages


# ------------------------------------------------------------------ estimation


def test_estimation_ordre_de_grandeur():
    """~3,5 caractères par token en français : c'est un repli, pas une mesure."""
    assert _jetons_estimes("") == 1
    assert _jetons_estimes("a" * 350) == 101
    assert _jetons_estimes("a" * 3500) == 1001


def test_un_texte_plus_long_est_toujours_estime_plus_gros():
    assert _jetons_estimes("court") < _jetons_estimes("beaucoup plus long que le premier")


def test_limite_est_quatre_vingts_pour_cent_du_contexte():
    """Les 20 % restants sont pour la réponse, dont on ignore la longueur."""
    assert session(n_ctx=10_000)._limite_jetons() == 8_000
    assert session(n_ctx=32_768)._limite_jetons() == 26_214


def test_sans_contexte_annonce_la_limite_est_nulle():
    """Contexte inconnu : on ne peut pas compter en tokens, le repli prend la main."""
    assert session(n_ctx=0)._limite_jetons() == 0


# --------------------------------------------------------------------- ancrage


def test_sans_ancrage_tout_est_estime():
    s = session()
    s.messages = echanges(2)
    attendu = sum(_jetons_estimes(m["content"]) for m in s.messages) + 4 * len(s.messages)
    assert s._jetons() == attendu


def test_avec_ancrage_seul_le_delta_est_estime():
    """Le cœur du mécanisme : on part du compte exact du serveur, on n'estime que la suite."""
    s = session()
    s.messages = echanges(2)  # 1 message système + 2 échanges = 5 messages
    s._ancrage_jetons = 1_234  # ce que le serveur a compté pour ces 5 messages
    s._ancrage_messages = len(s.messages)
    s.messages += [
        {"role": "user", "content": "nouvelle question"},
        {"role": "assistant", "content": "nouvelle réponse"},
    ]
    delta = _jetons_estimes("nouvelle question") + _jetons_estimes("nouvelle réponse")
    assert s._jetons() == 1_234 + delta


def test_ancrage_ignore_si_des_messages_ont_ete_retires():
    """Sinon on compterait un historique qui n'existe plus."""
    s = session()
    s.messages = echanges(1)  # 3 messages seulement
    s._ancrage_jetons = 5_000
    s._ancrage_messages = 9  # ancrage d'avant un élagage : inutilisable
    attendu = sum(_jetons_estimes(m["content"]) for m in s.messages) + 4 * len(s.messages)
    assert s._jetons() == attendu


# ------------------------------------------------------------------- découpage


def test_elagage_respecte_le_budget_de_tokens():
    # 12 échanges de 400 caractères par message pèsent ~2 900 tokens : le budget de
    # 2 000 (80 % de 2 500) doit donc déclencher l'élagage.
    s = session(n_ctx=2_500)
    s.messages = echanges(12)
    assert s._jetons() > s._limite_jetons()
    s._trim_history()
    assert s._jetons() <= s._limite_jetons()


def test_elagage_garde_le_message_systeme():
    s = session(n_ctx=2_000)
    s.messages = echanges(10)
    s._trim_history()
    assert s.messages[0]["role"] == "system"
    assert s.messages[0]["content"] == "Tu es concis."


def test_elagage_retire_des_echanges_complets():
    """L'historique ne doit pas commencer sur une réponse orpheline."""
    s = session(n_ctx=2_000)
    s.messages = echanges(10)
    s._trim_history()
    assert s.messages[1]["role"] == "user"


def test_elagage_ne_vide_jamais_tout():
    """Même avec un contexte minuscule, il reste de quoi répondre."""
    s = session(n_ctx=50)  # budget ridicule : 40 tokens
    s.messages = echanges(6)
    s._trim_history()
    assert len(s.messages) >= 3


def test_elagage_invalide_l_ancrage():
    """Le compte exact du serveur ne décrit plus la nouvelle liste."""
    s = session(n_ctx=2_000)
    s.messages = echanges(10)
    s._ancrage_jetons = 9_999
    s._ancrage_messages = 5
    s._trim_history()
    assert s._ancrage_jetons == 0


def test_sans_contexte_retombe_sur_le_nombre_de_messages():
    """Serveur muet : l'ancien comportement, et c'est dit dans l'avertissement."""
    s = session(n_ctx=0, history_limit=4)
    s.messages = echanges(10)
    s._trim_history()
    assert len(s.messages) == 5  # le système + les 4 derniers
    assert s.messages[0]["role"] == "system"


def test_un_historique_raisonnable_n_est_pas_touche():
    """Le cas normal : rien n'est retiré, et surtout rien n'est annoncé."""
    s = session(n_ctx=32_768)
    s.messages = echanges(3)
    avant = list(s.messages)
    s._trim_history()
    assert s.messages == avant


# ----------------------------------------------------------------- avertissements


def test_l_oubli_est_annonce_une_seule_fois(capsys):
    s = session(n_ctx=2_000)
    s.messages = echanges(10)
    s._prevenir_oubli(2, "limite : 1 600 tokens")
    s._prevenir_oubli(2, "limite : 1 600 tokens")
    sortie = capsys.readouterr().out
    assert sortie.count("ne sont plus envoyés") == 1
    assert "/resume" in sortie


def test_l_elagage_annonce_ce_qui_sort(capsys):
    s = session(n_ctx=2_000)
    s.messages = echanges(10)
    s._trim_history()
    sortie = capsys.readouterr().out
    assert "ne sont plus envoyés" in sortie
    assert "tokens" in sortie


def test_reset_remet_les_compteurs_a_zero():
    s = session()
    s._oubli_signale = True
    s._contexte_signale = True
    s._ancrage_jetons = 500
    # Ce que fait /reset
    s.messages = [{"role": "system", "content": "Tu es concis."}]
    s._oubli_signale = False
    s._contexte_signale = False
    s._ancrage_jetons = 0
    s._ancrage_messages = 0
    assert s._jetons() < 20


# ------------------------------------------------------------ alerte des 80 %


class FauxUsage:
    def __init__(self, prompt: int, completion: int = 0) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


def test_alerte_quand_le_contexte_approche(capsys):
    s = session(n_ctx=1_000)
    s._avis_contexte(FauxUsage(850, 50))
    sortie = capsys.readouterr().out
    assert "900 / 1000" in sortie
    assert "/resume" in sortie


def test_pas_d_alerte_quand_il_rest_de_la_place(capsys):
    s = session(n_ctx=32_768)
    s._avis_contexte(FauxUsage(1_646, 120))
    assert "contexte]" not in capsys.readouterr().out


def test_l_alerte_ne_se_repete_pas(capsys):
    s = session(n_ctx=1_000)
    s._avis_contexte(FauxUsage(900))
    s._avis_contexte(FauxUsage(950))
    assert capsys.readouterr().out.count("/resume") == 1


def test_pas_d_alerte_sans_contexte_connu(capsys):
    s = session(n_ctx=0)
    s._avis_contexte(FauxUsage(100_000))
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------- /contexte


def test_contexte_affiche_les_chiffres(capsys):
    s = session(n_ctx=10_000)
    s.messages = echanges(2)
    s._montrer_contexte()
    sortie = capsys.readouterr().out
    assert "10 000" in sortie or "10000" in sortie
    assert "80 %" in sortie


def test_contexte_sans_serveur_muet_ne_plante_pas(capsys):
    session(n_ctx=0)._montrer_contexte()
    assert "le serveur ne l'annonce pas" in capsys.readouterr().out


def test_contexte_signale_quand_c_est_plein(capsys):
    s = session(n_ctx=1_000)
    s.messages = echanges(5)  # bien au-delà du budget
    s._montrer_contexte()
    assert "/resume" in capsys.readouterr().out


# ------------------------------------------------------- le cas réel mesuré


def test_une_conversation_de_la_taille_mesuree_est_loin_de_la_limite():
    """25 messages ≈ 1 646 tokens réels : on doit être à ~5 % du contexte de 32 768.

    C'est la mesure qui justifie tout ce fichier : l'ancienne limite de 24 messages
    coupait alors que 95 % du contexte était libre.
    """
    s = session(n_ctx=32_768)
    messages = [{"role": "system", "content": "Tu es un assistant francophone concis et précis."}]
    question = "Peux-tu m expliquer comment fonctionne la réserve audio du projet ?"
    reponse = "La réserve garde le son des phrases déjà synthétisées. " * 3  # ~250 caractères
    for _ in range(12):
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": reponse})
    s.messages = messages
    estimation = s._jetons()
    assert estimation < 0.10 * 32_768, f"estimation trop haute : {estimation}"
    s._trim_history()
    assert s.messages == messages, "rien ne doit être retiré à ce niveau de remplissage"
