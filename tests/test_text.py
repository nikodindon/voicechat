"""Tests du découpage en phrases et du nettoyage TTS (sans réseau ni audio)."""

from voicechat.text import clean_for_speech, split_sentences


def test_coupe_sur_ponctuation_forte():
    phrases, reste = split_sentences("Bonjour. Comment vas-tu ? ")
    assert phrases == ["Bonjour.", "Comment vas-tu ?"]
    assert reste == ""


def test_garde_le_fragment_incomplet():
    phrases, reste = split_sentences("Le chat dort. Il rêve d'une")
    assert phrases == ["Le chat dort."]
    assert reste == "Il rêve d'une"


def test_point_decimal_non_coupe():
    phrases, _ = split_sentences("Pi vaut 3.14 environ.")
    assert phrases == ["Pi vaut 3.14 environ."]


def test_abreviation_non_coupee():
    phrases, reste = split_sentences("M. Dupont arrive demain. ")
    assert phrases == ["M. Dupont arrive demain."]
    assert reste == ""


def test_saut_de_ligne_coupe():
    phrases, _ = split_sentences("Première ligne\nDeuxième ligne\n")
    assert phrases == ["Première ligne", "Deuxième ligne"]


def test_segment_trop_long_decoupe():
    long = "mot " * 100  # 400 caractères sans ponctuation
    phrases, _ = split_sentences(long + ".", max_chars=100)
    assert phrases, "au moins un segment attendu"
    assert all(len(p) <= 100 for p in phrases)


def test_miette_ignoree():
    phrases, _ = split_sentences("Bonjour. !!! Salut.")
    assert all(len(p) >= 2 for p in phrases)
    assert "Bonjour." in phrases


def test_reste_sans_ponctuation_finit_par_etre_emis():
    """Un modèle qui n'écrit aucun point ne doit pas bloquer la voix indéfiniment."""
    tampon = "mot " * 200  # 800 caractères, zéro ponctuation
    phrases, reste = split_sentences(tampon, max_chars=100)
    assert phrases, "le reste doit être émis par morceaux, pas accumulé"
    assert all(len(p) <= 100 for p in phrases)
    assert len(reste) <= 100


def test_mot_geant_sans_espace_nest_pas_emiette():
    """Un unique « mot » très long ne peut pas être coupé : on attend la suite."""
    phrases, reste = split_sentences("a" * 300, max_chars=100)
    assert phrases == []
    assert len(reste) == 300


def test_flux_progressif():
    """Le cas réel : les morceaux arrivent un par un depuis le LLM."""
    tampon = ""
    phrases: list[str] = []
    for morceau in ["Voici ", "une ", "phrase. ", "Puis ", "une autre", ".", " Et fin."]:
        tampon += morceau
        nouveau, tampon = split_sentences(tampon)
        phrases.extend(nouveau)
    if tampon.strip():
        phrases.append(tampon.strip())
    assert phrases == ["Voici une phrase.", "Puis une autre.", "Et fin."]


def test_nettoyage_markdown():
    assert clean_for_speech("**gras** et `code`") == "gras et code"
    assert clean_for_speech("- item") == "item"


def test_nettoyage_bloc_de_code():
    sortie = clean_for_speech("Voici :\n```python\nprint(1)\n```\nTerminé.")
    assert "print" not in sortie
    assert "Terminé." in sortie
