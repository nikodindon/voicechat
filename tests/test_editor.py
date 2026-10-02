"""Tests du lecteur clavier, pilotés via un vrai pseudo-terminal (pty).

Ces tests reproduisent fidèlement ce que fait un terminal : ils écrivent des octets
sur le maître du pty, exactement comme le presse-papiers le ferait, et vérifient ce
que ``LineEditor.read()`` retourne.

Note sur pytest : son système de capture réinstalle son propre ``sys.stdout`` au
passage en phase « call ». Un patch de ``sys.stdout`` posé dans une **fixture** est
donc écrasé ; il doit être posé dans le corps du test (c'est ce que fait ``jouer``).
``sys.stdin`` n'est pas concerné et reste patchable en fixture.
"""

from __future__ import annotations

import io
import os
import pty
import sys
import threading
import time
import tty

import pytest

from voicechat.editor import (
    PASTE_END,
    PASTE_START,
    ClavierGeneration,
    LineEditor,
    _est_prefixe,
    _jeton,
)

DELAI = 0.06  # s entre deux « évènements clavier »


class _StdinFactice:
    """Fait passer un descripteur de pty pour un stdin interactif."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd


@pytest.fixture()
def pty_env(monkeypatch):
    maitre, esclave = pty.openpty()
    tty.setraw(esclave)  # comme le fait le lecteur : les octets passent tels quels
    monkeypatch.setattr(sys, "stdin", _StdinFactice(esclave))
    yield maitre, esclave
    os.close(maitre)
    os.close(esclave)


def jouer(maitre, editeur, ecritures, sortie=None, initial=""):
    """Lance read() en tâche de fond et envoie les octets un par un.

    Chaque entrée de ``ecritures`` est un évènement distinct (une frappe, un collage,
    une touche), séparé du suivant par un délai — comme dans la réalité.
    Retourne le message lu ; ``sortie`` (si fourni) reçoit l'affichage.
    """
    if sortie is None:
        sortie = io.StringIO()
    resultat: dict[str, str | None] = {}

    def cible():
        resultat["valeur"] = editeur.read(initial=initial)

    ancien_stdout = sys.stdout
    sys.stdout = sortie  # posé ici, donc pendant la phase « call » : il tient
    try:
        fil = threading.Thread(target=cible, daemon=True)
        fil.start()
        time.sleep(DELAI)  # laisse le temps au lecteur d'entrer dans sa boucle
        for octets in ecritures:
            os.write(maitre, octets)
            time.sleep(DELAI)
        fil.join(timeout=5)
        assert not fil.is_alive(), "read() ne s'est jamais terminé"
    finally:
        sys.stdout = ancien_stdout
    return resultat.get("valeur")


# ------------------------------------------------------------------ découpage pur
def test_jeton_reconnait_les_sequences():
    assert _jeton(PASTE_START + "abc") == (PASTE_START, "abc")
    assert _jeton("\x1b[A") == ("\x1b[A", "")
    assert _jeton("a") == ("a", "")


def test_jeton_attend_une_sequence_incomplete():
    jeton, reste = _jeton("\x1b[20")
    assert jeton is None, "une séquence partielle ne doit pas être interprétée"
    assert reste == "\x1b[20"


def test_est_prefixe():
    assert _est_prefixe("\x1b[")
    assert _est_prefixe("\x1b[200")
    assert not _est_prefixe("x")


# ------------------------------------------------------------------ collage
def test_collage_balise_devient_un_seul_message(pty_env):
    """Le cas rapporté : un texte collé de 3 lignes = UN message, envoyé par Entrée."""
    maitre, _ = pty_env
    texte = "Première ligne du texte collé.\nDeuxième ligne.\nTroisième ligne."
    resultat = jouer(
        maitre,
        LineEditor(),
        [f"{PASTE_START}{texte}{PASTE_END}".encode(), b"\r"],
    )
    assert resultat == texte
    assert resultat.count("\n") == 2, "les sauts de ligne doivent être conservés"


def test_collage_ne_declenche_pas_d_envoi(pty_env):
    """Après un collage, rien n'est envoyé tant que Entrée n'a pas été frappée."""
    maitre, _ = pty_env
    editeur = LineEditor()
    resultat: dict[str, str | None] = {}
    sortie = io.StringIO()

    def cible():
        resultat["valeur"] = editeur.read()

    ancien = sys.stdout
    sys.stdout = sortie
    try:
        fil = threading.Thread(target=cible, daemon=True)
        fil.start()
        time.sleep(DELAI)
        os.write(maitre, f"{PASTE_START}a\nb\nc{PASTE_END}".encode())
        time.sleep(DELAI * 3)
        assert fil.is_alive(), "un collage ne doit PAS envoyer le message"
        os.write(maitre, b"\r")  # c'est seulement maintenant qu'on envoie
        fil.join(timeout=5)
    finally:
        sys.stdout = ancien
    assert resultat["valeur"] == "a\nb\nc"


def test_collage_sans_balise_detecte_par_rafale(pty_env):
    """Repli : terminal sans bracketed paste — la rafale suffit à reconnaître un collage."""
    maitre, _ = pty_env
    resultat = jouer(
        maitre,
        LineEditor(),
        [b"ligne un\nligne deux\nligne trois", b"\r"],
    )
    assert resultat == "ligne un\nligne deux\nligne trois"


def test_une_seule_ligne_collee_reste_normale(pty_env):
    maitre, _ = pty_env
    resultat = jouer(maitre, LineEditor(), [b"juste une ligne", b"\r"])
    assert resultat == "juste une ligne"


def test_frappe_lente_reste_un_seul_message(pty_env):
    """Taper caractère par caractère ne doit jamais ressembler à un collage."""
    maitre, _ = pty_env
    resultat = jouer(maitre, LineEditor(), [b"b", b"o", b"n", b"j", b"o", b"u", b"r", b"\r"])
    assert resultat == "bonjour"


# ------------------------------------------------------------------ touches
def test_retour_arriere(pty_env):
    maitre, _ = pty_env
    resultat = jouer(maitre, LineEditor(), [b"bonjourr", b"\x7f", b"\r"])
    assert resultat == "bonjour"


def test_ctrl_u_efface_le_brouillon(pty_env):
    maitre, _ = pty_env
    resultat = jouer(maitre, LineEditor(), [b"a effacer", b"\x15", b"propre", b"\r"])
    assert resultat == "propre"


def test_entree_sur_ligne_vide(pty_env):
    maitre, _ = pty_env
    assert jouer(maitre, LineEditor(), [b"\r"]) == ""


def test_ctrl_d_sur_brouillon_vide_termine(pty_env):
    maitre, _ = pty_env
    assert jouer(maitre, LineEditor(), [b"\x04"]) is None


def test_historique_fleche_haut(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor(history_limit=10)
    editeur.history = ["message precedent"]
    resultat = jouer(maitre, editeur, [b"\x1b[A", b"\r"])
    assert resultat == "message precedent"


def test_historique_alimente_apres_envoi(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor()
    jouer(maitre, editeur, [b"souvenir", b"\r"])
    assert editeur.history == ["souvenir"]


def test_affichage_prompt_et_curseur(pty_env):
    """L'affichage doit contenir le prompt et replacer le curseur après édition."""
    maitre, _ = pty_env
    sortie = io.StringIO()
    resultat = jouer(maitre, LineEditor(), [b"abc", b"\x1b[D", b"\x1b[D", b"X", b"\r"], sortie=sortie)
    assert resultat == "aXbc", "les flèches gauche doivent déplacer le curseur"
    affichage = sortie.getvalue()
    assert "vous › " in affichage
    assert "\x1b[1D" in affichage, "le curseur doit être replacé vers la gauche"
    assert "\x1b[?2004h" in affichage, "le bracketed paste doit être activé"
    assert "\x1b[?2004l" in affichage, "et désactivé à la sortie"


def test_affichage_resume_pour_texte_long(pty_env):
    """Au-delà d'une ligne écran, on résume au lieu de tout redessiner."""
    maitre, _ = pty_env
    sortie = io.StringIO()
    texte = "\n".join(f"ligne {i}" for i in range(12))
    jouer(maitre, LineEditor(), [f"{PASTE_START}{texte}{PASTE_END}".encode(), b"\r"], sortie=sortie)
    affichage = sortie.getvalue()
    assert "12 lignes" in affichage
    assert "⏎ envoyer" in affichage


# --------------------------------------------------- entrée non interactive
def test_entree_pipee_une_ligne_par_message(monkeypatch):
    """Un script qui écrit plusieurs lignes attend plusieurs tours, pas un collage."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("premier\nsecond\n"))
    editeur = LineEditor()
    assert editeur.read() == "premier"
    assert editeur.read() == "second"
    assert editeur.read() is None  # fin de flux


def test_entree_pipee_garde_les_lignes_vides(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\n"))
    editeur = LineEditor()
    assert editeur.read() == ""
    assert editeur.read() == ""
    assert editeur.read() is None


def test_brouillon_prerempli(pty_env):
    """Les frappes faites pendant une réponse doivent revenir au prompt suivant."""
    maitre, _ = pty_env
    resultat = jouer(maitre, LineEditor(), [b" ajoute", b"\r"], initial="deja la")
    assert resultat == "deja la ajoute"


# ------------------------------------- clavier pendant une génération (Échap)
def test_clavier_generation_detecte_echap(pty_env):
    maitre, _ = pty_env
    with ClavierGeneration() as clavier:
        os.write(maitre, b"\x1b")     # Échap seul
        time.sleep(0.02)
        assert clavier.sonder() is False, "trop tôt : la séquence peut encore s'allonger"
        time.sleep(0.08)              # au-delà de ESC_TIMEOUT
        assert clavier.sonder() is True


def test_echap_est_signale_une_seule_fois(pty_env):
    maitre, _ = pty_env
    with ClavierGeneration() as clavier:
        os.write(maitre, b"\x1b")
        time.sleep(0.02)
        clavier.sonder()
        time.sleep(0.08)
        assert clavier.sonder() is True
        assert clavier.sonder() is False, "le signal doit être à front, pas à niveau"


def test_fleche_nest_pas_prise_pour_echap(pty_env):
    """Une flèche commence aussi par \\x1b : elle ne doit pas couper la voix."""
    maitre, _ = pty_env
    with ClavierGeneration() as clavier:
        os.write(maitre, b"\x1b[A")
        time.sleep(0.08)
        assert clavier.sonder() is False


def test_frappes_conservees_pendant_la_generation(pty_env):
    maitre, _ = pty_env
    with ClavierGeneration() as clavier:
        os.write(maitre, b"salut")
        time.sleep(0.05)
        clavier.sonder()
    assert clavier.tampon == "salut"


def test_echap_et_frappes_melanges(pty_env):
    """Échap n'efface pas le texte déjà tapé, et le texte n'empêche pas Échap."""
    maitre, _ = pty_env
    with ClavierGeneration() as clavier:
        os.write(maitre, b"abc")
        time.sleep(0.05)
        clavier.sonder()
        os.write(maitre, b"\x1b")
        time.sleep(0.02)
        clavier.sonder()
        time.sleep(0.08)
        assert clavier.sonder() is True
    assert clavier.tampon == "abc"


def test_clavier_generation_sans_tty_ne_plante_pas(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with ClavierGeneration() as clavier:
        assert clavier.sonder() is False


# ------------------------------------------------- recherche dans l'historique
def test_recherche_ctrl_r(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor()
    editeur.history = ["le chat dort", "le chien aboie", "autre chose"]
    assert jouer(maitre, editeur, [b"\x12", b"chien", b"\r"]) == "le chien aboie"


def test_recherche_ctrl_r_descend_vers_le_plus_ancien(pty_env):
    """Ctrl+R relancé doit trouver une occurrence plus ancienne, pas rester sur place."""
    maitre, _ = pty_env
    editeur = LineEditor()
    editeur.history = ["le chat", "le chien", "le cheval"]
    assert jouer(maitre, editeur, [b"\x12", b"le ch", b"\x12", b"\r"]) == "le chien"


def test_recherche_ctrl_g_restaure_le_brouillon(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor()
    editeur.history = ["trouve moi ca"]
    assert jouer(maitre, editeur, [b"brouillon", b"\x12", b"trouve", b"\x07", b"\r"]) == "brouillon"


def test_recherche_sans_correspondance_laisse_le_brouillon(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor()
    editeur.history = ["quelque chose"]
    assert jouer(maitre, editeur, [b"abc", b"\x12", b"zzz", b"\r"]) == "abc"


def test_recherche_retour_arriere_raccourcit_le_motif(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor()
    editeur.history = ["pomme", "poire"]
    # « poi » ne trouve rien, mais après un retour arrière « po » trouve « poire »
    assert jouer(maitre, editeur, [b"\x12", b"poiz", b"\x7f", b"\r"]) == "poire"


def test_recherche_historique_vide_ne_plante_pas(pty_env):
    """Sans historique, la recherche ne trouve rien — comme readline — mais ne casse pas."""
    maitre, _ = pty_env
    assert jouer(maitre, LineEditor(), [b"\x12", b"abc", b"\r"]) == ""


def test_recherche_motif_vide_naffiche_pas_de_resultat(pty_env):
    """Ctrl+R seul affiche l'invite de recherche, sans vider le brouillon."""
    maitre, _ = pty_env
    editeur = LineEditor()
    editeur.history = ["quelque chose"]
    sortie = io.StringIO()
    assert jouer(maitre, editeur, [b"\x12", b"\x07", b"\r"], sortie=sortie) == ""
    assert "recherche inversée" in sortie.getvalue()


# ------------------------------------------------------------ complétion (Tab)
def test_completion_commande_unique(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor(completeur=lambda t: ["/quit", "/voice"])
    assert jouer(maitre, editeur, [b"/qu", b"\t", b"\r"]) == "/quit"


def test_completion_prefixe_commun(pty_env):
    """Deux candidats : on complète jusqu'au plus long préfixe commun."""
    maitre, _ = pty_env
    editeur = LineEditor(completeur=lambda t: ["/voice", "/voices"])
    assert jouer(maitre, editeur, [b"/vo", b"\t", b"\r"]) == "/voice"


def test_completion_argument(pty_env):
    maitre, _ = pty_env

    def comp(avant: str) -> list[str]:
        if avant.startswith("/voice "):
            return ["af_heart", "ff_siwis", "if_sara"]
        return ["/voice"]

    editeur = LineEditor(completeur=comp)
    assert jouer(maitre, editeur, [b"/voice f", b"\t", b"\r"]) == "/voice ff_siwis"


def test_completion_sans_candidat_ne_touche_a_rien(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor(completeur=lambda t: ["/quit"])
    assert jouer(maitre, editeur, [b"/xyz", b"\t", b"\r"]) == "/xyz"


def test_completion_affiche_les_choix_quand_aucun_progres(pty_env):
    maitre, _ = pty_env
    editeur = LineEditor(completeur=lambda t: ["/voice", "/voices"])
    sortie = io.StringIO()
    # après la première complétion on est sur « /voice » : Tab ne peut plus avancer
    jouer(maitre, editeur, [b"/vo", b"\t", b"\t", b"\r"], sortie=sortie)
    assert "/voices" in sortie.getvalue()


def test_completeur_absent_ignore_la_touche_tab(pty_env):
    maitre, _ = pty_env
    assert jouer(maitre, LineEditor(), [b"/vo", b"\t", b"\r"]) == "/vo"


def test_completeur_en_erreur_ne_casse_pas_la_saisie(pty_env):
    """Un completeur qui lève ne doit pas faire perdre ce qui est tapé."""
    maitre, _ = pty_env

    def casse(_: str) -> list[str]:
        raise RuntimeError("boum")

    editeur = LineEditor(completeur=casse)
    assert jouer(maitre, editeur, [b"/vo", b"\t", b"\r"]) == "/vo"
