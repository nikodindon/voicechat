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


# ------------------------------------------------- liens, images, structure
def test_lien_markdown_garde_le_texte_pas_lurl():
    """Régression : l'ancien code retirait l'URL et laissait « [texte]( » à lire."""
    sortie = clean_for_speech("Voir [ce guide](https://exemple.fr/doc) pour la suite.")
    assert sortie == "Voir ce guide pour la suite."
    assert "[" not in sortie and "]" not in sortie and "(" not in sortie


def test_image_markdown_retiree_entierement():
    assert clean_for_speech("![capture](img.png) fin") == "fin"
    assert "capture" not in clean_for_speech("![capture](img.png) fin")


def test_url_nue_devient_lien():
    assert clean_for_speech("Voir https://exemple.fr/page") == "Voir lien"


def test_titre_et_citation_nettoyes():
    assert clean_for_speech("### Mon titre") == "Mon titre"
    assert clean_for_speech("> une citation") == "une citation"


def test_puces_et_listes_numerotees():
    assert clean_for_speech("- un point") == "un point"
    assert clean_for_speech("* un point") == "un point"
    assert clean_for_speech("— un point") == "un point"
    assert clean_for_speech("1. premier point") == "premier point"
    assert clean_for_speech("2) second point") == "second point"


def test_tableau_lu_cellule_par_cellule():
    assert clean_for_speech("| Réglage | Gain |") == "Réglage, Gain"


def test_separateur_de_tableau_et_regle_horizontale_ignores():
    assert clean_for_speech("|---|---|") == ""
    assert clean_for_speech("---") == ""


def test_emoji_retires():
    assert clean_for_speech("Bravo ! 😀🎉") == "Bravo !"


# ------------------------------------------------- unités et abréviations FR
def test_unites_prononcables():
    assert clean_for_speech("30 %") == "30 pour cent"
    assert clean_for_speech("32 °C") == "32 degrés"
    assert clean_for_speech("15 €") == "15 euros"
    assert clean_for_speech("90 km/h") == "90 kilomètres par heure"


def test_abreviations_developpees():
    assert clean_for_speech("M. Dupont") == "Monsieur Dupont"
    assert clean_for_speech("Mme Martin") == "Madame Martin"
    assert clean_for_speech("le Dr House") == "le Docteur House"
    assert clean_for_speech("n° 12") == "numéro 12"


def test_sigle_ne_mange_pas_un_mot_plus_long():
    """« Pr » ne doit pas transformer « Premier » en « Professeuremier »."""
    assert clean_for_speech("Premier point") == "Premier point"
    assert clean_for_speech("une enveloppe") == "une enveloppe"


def test_milliers_sans_espaces():
    assert clean_for_speech("1 000 tours") == "1000 tours"
    assert clean_for_speech("1\u00a0500 euros") == "1500 euros"


# ------------------------------------------------- blocs de code et le découpeur
def test_bloc_de_code_jamais_prononce():
    phrases, _ = split_sentences("Texte.\n```bash\nls -l /tmp\n```\n")
    assert phrases == ["Texte."]
    assert "ls" not in " ".join(phrases)


def test_bloc_de_code_ouvert_est_retenu_pas_prononce():
    """Tant que le bloc n'est pas refermé, son contenu attend au lieu d'être lu."""
    phrases, reste = split_sentences("Voici :\n```bash\nls -l\n")
    assert phrases == ["Voici :"]
    assert "ls -l" in reste


def test_bloc_de_code_ferme_rend_la_suite_prononcable():
    phrases, _ = split_sentences("Voici :\n```bash\nls -l\n```\nLa suite.\n")
    assert phrases == ["Voici :", "La suite."]


def test_backtick_isole_nest_pas_prononce():
    """En flux, les ``` arrivent un par un : un « ` » seul ne doit pas partir au TTS."""
    phrases, reste = split_sentences("Fin du code.\n`")
    assert phrases == ["Fin du code."]
    assert reste.endswith("`")


def test_marqueur_de_liste_ne_coupe_pas_la_phrase():
    """Régression : « 2. » partait seul à la synthèse et l'auditeur entendait « deux »."""
    phrases, _ = split_sentences("- 2. Réduis la charge CPU.\n")
    assert phrases == ["- 2. Réduis la charge CPU."]
    assert clean_for_speech(phrases[0]) == "Réduis la charge CPU."


def test_abreviation_en_fin_de_tampon_ne_coupe_pas():
    """Régression : le paquet SSE s'arrête pile après « M. » et « Dupont » partait seul.

    L'auditeur entendait alors « chapitre de M. » puis « Dupont. », l'abréviation
    n'était plus développée, et la phrase était coupée en deux.
    """
    tampon = ""
    phrases: list[str] = []
    for morceau in ["Voir le chapitre de ", "M. ", "Dupont."]:
        tampon += morceau
        nouveau, tampon = split_sentences(tampon)
        phrases.extend(clean_for_speech(p) for p in nouveau)
    if tampon.strip():
        phrases.append(clean_for_speech(tampon))
    assert phrases == ["Voir le chapitre de Monsieur Dupont."]
