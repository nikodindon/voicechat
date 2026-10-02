"""Tests du mode dialogue (v1.5) : deux personas, deux voix, une alternance.

Aucun GPU, aucun modèle : la session est une doublure qui note ce qu'on lui demande. Ce
qui est éprouvé ici, c'est **l'ordre des opérations** — parce que c'est là que le mode
dialogue peut se tromper de voix sans que rien ne le signale, et que le texte, lui, reste
parfaitement correct.
"""

from __future__ import annotations

import pytest

from voicechat import store
from voicechat.cli import Tour
from voicechat.dialogue import CONSIGNE, EN_FACE, Dialogue, Persona
from voicechat.llm import Usage


# ------------------------------------------------------------------ les doublures
class FauxTTS:
    """Assez de TTS pour que le dialogue s'y croie."""

    def __init__(self, voice: str = "ff_siwis", speed: float = 1.0) -> None:
        self.voice = voice
        self.speed = speed
        self.lang_code = "f"

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    def set_lang(self, lang: str) -> None:
        self.lang_code = lang


class FauxSpeech:
    """Note les vidages de file : c'est **avant** la bascule qu'ils doivent tomber."""

    def __init__(self, journal: list[str]) -> None:
        self.journal = journal

    def wait(self) -> None:
        self.journal.append("attente-file")


class FausseSession:
    """Tient lieu de ChatSession : répond un texte convenu, et journalise les appels."""

    def __init__(self, voix: str = "ff_siwis", reponses: list[str] | None = None) -> None:
        from voicechat.config import Config

        self.journal: list[str] = []
        self.cfg = Config(model="m.gguf", voice=voix, tts=False)
        self.tts = FauxTTS(voix)
        self.speech = FauxSpeech(self.journal)
        self.reponses = reponses or []
        self.tours: list[dict] = []

    def appliquer_reglages(self, profil: store.Profil) -> None:
        self.journal.append(f"reglages:{profil.voix or ''}")
        if profil.voix:
            self.tts.voice = profil.voix
            self.cfg.voice = profil.voix

    def tour_modele(self, messages, etiquette="ia › ", modele=None) -> Tour:
        self.tours.append(
            {
                "etiquette": etiquette,
                "modele": modele,
                "messages": list(messages),
                "voix": self.tts.voice,
            }
        )
        texte = self.reponses.pop(0) if self.reponses else f"réplique {len(self.tours)}"
        if texte == "":
            return Tour(reponse="", usage=Usage())
        messages.append({"role": "assistant", "content": texte})
        return Tour(reponse=texte, usage=Usage(prompt_tokens=100))


def _deux_personas(tmp_path, voix_a: str | None = "ff_siwis", voix_b: str | None = "ef_dora"):
    """Écrit deux profils dans un dossier isolé et les charge."""
    dossier = tmp_path / "profils"
    dossier.mkdir(exist_ok=True)
    (dossier / "alice.md").write_text(
        (f"voix: {voix_a}\n\n" if voix_a else "") + "Tu es Alice, tu aimes les idées nettes.",
        encoding="utf-8",
    )
    (dossier / "bob.md").write_text(
        (f"voix: {voix_b}\n\n" if voix_b else "") + "Tu es Bob, tu aimes les exemples.",
        encoding="utf-8",
    )
    monkeypatche = pytest.MonkeyPatch()
    monkeypatche.setenv("VOICECHAT_PROFILS", str(dossier))
    return Persona.charger("alice"), Persona.charger("bob")


# ---------------------------------------------------------------- les personas
def test_un_persona_charge_son_profil(tmp_path):
    a, b = _deux_personas(tmp_path)
    assert a.nom == "alice"
    assert a.voix == "ff_siwis"
    assert a.messages[0]["role"] == "system"
    assert "Alice" in a.messages[0]["content"]
    assert b.voix == "ef_dora"
    # Chacun sa liste : c'est ce qui permet de réutiliser le chemin normal d'un tour.
    assert a.messages is not b.messages


def test_un_persona_sans_prompt_est_refuse(tmp_path):
    dossier = tmp_path / "profils"
    dossier.mkdir()
    (dossier / "vide.md").write_text("voix: ff_siwis\n", encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("VOICECHAT_PROFILS", str(dossier))
    with pytest.raises(ValueError, match="aucun prompt"):
        Persona.charger("vide")


def test_un_persona_inconnu(tmp_path):
    _deux_personas(tmp_path)
    with pytest.raises(FileNotFoundError):
        Persona.charger("personne")


# ------------------------------------------------------------------ l'alternance
def test_les_deux_personas_alternent(tmp_path):
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1", "A2", "B2"])
    dialogue = Dialogue(session, a, b, "le sens de la vie", tours=4)

    repliques = dialogue.derouler()

    assert [nom for nom, _ in repliques] == ["alice", "bob", "alice", "bob"]
    assert [texte for _, texte in repliques] == ["A1", "B1", "A2", "B2"]
    # Le premier reçoit la consigne (avec le sujet), les suivants reçoivent la réplique
    # précédente telle quelle — c'est le prompt système qui dit de qui elle vient.
    assert session.tours[0]["messages"][-1]["content"] == CONSIGNE.format(
        sujet="le sens de la vie"
    )
    assert session.tours[1]["messages"][-1]["content"] == "A1"
    assert session.tours[3]["messages"][-1]["content"] == "A2"


def test_chaque_persona_garde_sa_propre_histoire(tmp_path):
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1", "A2", "B2"])
    Dialogue(session, a, b, "sujet", tours=4).derouler()

    # Alice : son prompt, la consigne, sa réplique, puis celle de Bob, puis la sienne.
    assert [m["role"] for m in a.messages] == ["system", "user", "assistant", "user", "assistant"]
    assert a.messages[1]["content"].startswith("Sujet :")
    assert a.messages[3]["content"] == "B1"  # Bob lui parle…
    assert a.messages[4]["content"] == "A2"  # …elle répond
    # Bob ne voit jamais la consigne : il entre dans la conversation par la réplique
    # d'Alice. Et il sait qu'elle vient d'Alice — c'est écrit dans son prompt système.
    assert "Sujet :" not in b.messages[1]["content"]
    assert b.messages[1]["content"] == "A1"
    assert "Tu discutes avec alice" in b.messages[0]["content"]
    assert "Tu discutes avec bob" in a.messages[0]["content"]


def test_le_tour_recu_porte_le_nom_du_persona(tmp_path):
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1"])
    Dialogue(session, a, b, "sujet", tours=2).derouler()
    assert session.tours[0]["etiquette"] == "alice › "
    assert session.tours[1]["etiquette"] == "bob › "


def test_le_modele_du_profil_est_transmis(tmp_path):
    """Un persona peut tourner sur un autre modèle que la session."""
    a, b = _deux_personas(tmp_path)
    a.profil.modele = "petit.gguf"
    session = FausseSession(reponses=["A1", "B1"])
    Dialogue(session, a, b, "sujet", tours=2).derouler()
    assert session.tours[0]["modele"] == "petit.gguf"
    assert session.tours[1]["modele"] is None  # celui de la session


# --------------------------------------------------------------- le point sensible
def test_la_file_est_videe_avant_de_changer_de_voix(tmp_path):
    """L'ordre « vider puis basculer » est la seule chose qui empêche le mélange.

    Le thread de synthèse lit `tts.voice` au **dépilement** : si on bascule pendant que
    des phrases attendent, la persona suivante dit les répliques de la précédente.
    """
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1"])
    journal = session.journal

    Dialogue(session, a, b, "sujet", tours=2).derouler()

    # À chaque tour : attente de la file, puis réglages. Jamais l'inverse.
    for i, entree in enumerate(journal):
        if entree.startswith("reglages"):
            assert journal[i - 1] == "attente-file", f"bascule sans vidage : {journal}"
    # Deux bascules (une par persona)… plus le retour à la voix de la session, qui doit
    # lui aussi attendre : sinon la console se remettrait à parler avec la voix d'un
    # personnage au milieu d'une phrase.
    assert journal.count("attente-file") == 3
    assert journal.count("reglages:ff_siwis") == 1
    assert journal.count("reglages:ef_dora") == 1
    assert journal[-1] == "attente-file"


def test_chaque_replique_est_prononcee_avec_la_voix_de_son_persona(tmp_path):
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1", "A2", "B2"])
    Dialogue(session, a, b, "sujet", tours=4).derouler()
    # La voix au moment du tour doit être celle du persona qui parle.
    assert [t["voix"] for t in session.tours] == ["ff_siwis", "ef_dora", "ff_siwis", "ef_dora"]


def test_les_voix_effectives_repèrent_la_voix_absente(tmp_path):
    """Un profil sans `voix:` laisse celle de la session : les deux se ressembleraient."""
    a, b = _deux_personas(tmp_path, voix_a="ff_siwis", voix_b=None)
    session = FausseSession(voix="ff_siwis")
    assert Dialogue(session, a, b, "sujet", tours=2).voix_effectives() == (
        "ff_siwis",
        "ff_siwis",
    )

    # Avec deux voix déclarées, elles sont bien distinctes — c'est ce que `annoncer`
    # vérifie avant de commencer plutôt que de laisser découvrir le mélange à l'oreille.
    a2, b2 = _deux_personas(tmp_path, voix_a="ff_siwis", voix_b="ef_dora")
    assert Dialogue(session, a2, b2, "sujet", tours=2).voix_effectives() == (
        "ff_siwis",
        "ef_dora",
    )


def test_la_voix_de_la_session_est_rendue_a_la_fin(tmp_path):
    """Après un dialogue, la console ne doit pas garder la voix d'un personnage."""
    a, b = _deux_personas(tmp_path)
    session = FausseSession(voix="af_heart", reponses=["A1", "B1"])
    session.tts.voice = "af_heart"
    session.cfg.speed = 1.2
    session.cfg.lang = "a"

    Dialogue(session, a, b, "sujet", tours=2).derouler()

    assert session.tts.voice == "af_heart"
    assert session.cfg.voice == "af_heart"
    assert session.cfg.speed == 1.2
    assert session.tts.speed == 1.2


# ------------------------------------------------------------------ les arrêts
def test_une_interruption_arrete_le_dialogue(tmp_path):
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1"])
    dialogue = Dialogue(session, a, b, "sujet", tours=4)

    vrai = session.tour_modele

    def interrompu(messages, etiquette="ia › ", modele=None):
        tour = vrai(messages, etiquette, modele)
        if len(session.tours) == 2:
            tour.interrompu = True
        return tour

    session.tour_modele = interrompu  # type: ignore[method-assign]
    repliques = dialogue.derouler()

    # La réplique interrompue n'est pas gardée : elle est incomplète, et la compter
    # ferait croire à un tour qui a abouti.
    assert [nom for nom, _ in repliques] == ["alice"]
    assert len(session.tours) == 2, "on s'arrête au tour interrompu, sans relancer"


def test_une_reponse_vide_arrete_le_dialogue(tmp_path):
    """Panne réseau : insister ne sert à rien, le suivant n'aurait rien à répondre."""
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", ""])
    repliques = Dialogue(session, a, b, "sujet", tours=6).derouler()
    assert len(repliques) == 1
    assert len(session.tours) == 2  # on n'a pas relancé un troisième tour pour rien


def test_le_nombre_de_repliques_est_respecte(tmp_path):
    a, b = _deux_personas(tmp_path)
    session = FausseSession()
    assert len(Dialogue(session, a, b, "sujet", tours=3).derouler()) == 3


# ------------------------------------------------- les dérives de texte (à l'usage)
def test_une_replique_trop_longue_est_signalee(tmp_path, capsys):
    """Un dialogue s'écoute : 200 mots avec des titres, ça fait un rapport, pas une réplique."""
    a, b = _deux_personas(tmp_path)
    pave = "argument " * 120
    session = FausseSession(reponses=[pave, "court"])
    Dialogue(session, a, b, "sujet", tours=2).derouler()
    sortie = capsys.readouterr().out
    assert "longue à écouter" in sortie
    assert "120 mots" in sortie


def test_une_replique_courte_ne_declenche_rien(tmp_path, capsys):
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["Une réponse brève et nette.", "Une autre, tout aussi brève."])
    Dialogue(session, a, b, "sujet", tours=2).derouler()
    assert "longue à écouter" not in capsys.readouterr().out


def test_le_prompt_interdit_de_signier_son_nom(tmp_path):
    """La règle est dans le prompt système, pas dans le texte des messages.

    Deux versions précédentes ont échoué autrement : sans nom, le modèle s'appelait
    lui-même ; avec le nom en préfixe du message, il imitait le format et préfixait ses
    réponses par « bob : » — préfixe qui partait ensuite à la voix.
    """
    a, b = _deux_personas(tmp_path)
    session = FausseSession(reponses=["A1", "B1"])
    Dialogue(session, a, b, "sujet", tours=2).derouler()
    prompt_a = a.messages[0]["content"]
    assert "Tu discutes avec bob" in prompt_a
    assert "sans écrire « alice »" in prompt_a
    assert "sans recopier" in prompt_a
    # Et le texte des messages, lui, reste du texte : aucun préfixe de locuteur.
    assert session.tours[1]["messages"][-1]["content"] == "A1"
