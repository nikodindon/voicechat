"""Découpage du texte en phrases prononçables.

Fonctions pures, sans dépendance : testables hors ligne et réutilisables
quelle que soit la façon dont le texte arrive (streaming ou réponse complète).
"""

from __future__ import annotations

import re

# Fins de phrase : ponctuation forte, éventuellement suivie d'un guillemet/parenthèse,
# puis un espace ou la fin du tampon. Le saut de ligne termine aussi une phrase.
_SENT_RE = re.compile(r"[.!?…;:]+[»\"'\)\]]*(?:\s+|$)|\n+")

# Abréviations fréquentes : « M. Dupont » ne doit pas couper la phrase.
_ABBREV = (
    "m.", "mm.", "mme", "mr.", "mrs.", "dr.", "st.", "vs.", "etc.",
    "ex.", "cf.", "p.ex.", "env.", "av.", "ap.", "no.", "fig.", "tel.",
)

# Un point dans un nombre décimal (« 3.14 ») n'est pas une fin de phrase.
_NUMBER_RE = re.compile(r"\d\.\d$")

# Marqueur de liste numérotée (« 2. » en début de ligne) : le point n'est pas une
# fin de phrase, sinon « 2. » part seul à la synthèse et l'auditeur entend « deux ».
_LISTE_NUM_RE = re.compile(r"(?:^|\s)\d{1,2}\.$")


def _limite_code(buffer: str) -> int:
    """Position du ``` d'ouverture d'un bloc de code non refermé.

    Vaut la fin du tampon quand tous les blocs sont refermés. Tant qu'un bloc est
    ouvert, rien de ce qui le suit ne doit partir à la synthèse : on attend sa
    fermeture. Sinon le découpage par lignes hache la commande en morceaux et
    l'auditeur entend « bash, powermetrics, sensor ».
    """
    positions = [m.start() for m in re.finditer(r"```", buffer)]
    if len(positions) % 2:
        return positions[-1]
    # Un « ``` » qui arrive caractère par caractère : tant qu'il n'est pas complet il
    # ne compte pas comme délimiteur, mais les backticks déjà reçus ne doivent pas
    # partir à la synthèse (« ` » se prononce « accent grave » sur certains moteurs).
    partiel = len(buffer) - len(buffer.rstrip("`"))
    if partiel and partiel < 3:
        return len(buffer) - partiel
    return len(buffer)


def _dans_code(buffer: str, position: int) -> bool:
    """Vrai si ``position`` tombe à l'intérieur d'un bloc ``` … ```."""
    return buffer.count("```", 0, position) % 2 == 1


def split_sentences(
    buffer: str,
    max_chars: int = 220,
    min_chars: int = 2,
) -> tuple[list[str], str]:
    """Découpe ``buffer`` en phrases complètes.

    Retourne ``(phrases_prêtes_pour_le_tts, reste_incomplet)``.
    Le reste est un fragment sans ponctuation finale : il doit être renvoyé
    au prochain appel, une fois que la suite du flux est arrivée.

    ``max_chars`` borne la longueur d'un segment : au-delà on coupe au dernier
    espace, pour ne jamais envoyer 500 caractères d'un coup au synthétiseur.
    """
    if not buffer:
        return [], ""

    phrases: list[str] = []
    pos = 0
    limite = _limite_code(buffer)

    for match in _SENT_RE.finditer(buffer):
        end = match.end()
        if end > limite:
            break  # on ne franchit pas un bloc de code ouvert
        candidat = buffer[pos:end]
        if not candidat.strip():
            pos = end
            continue

        # La ligne de clôture « ``` » ne doit pas être prononcée non plus.
        if "```" in candidat or _dans_code(buffer, pos):
            pos = end
            continue

        nu = candidat.rstrip()
        bas = nu.lower()
        # Un marqueur de liste (« 2. »), un nombre décimal ou une abréviation connue
        # n'est jamais une fin de phrase — même si le tampon s'arrête pile là : en
        # flux, la suite n'a pas encore été reçue et on ne peut pas trancher
        # autrement. Le texte ainsi retenu n'est pas perdu : le CLI vide le tampon
        # à la fin de la réponse (cli.py, avant d'attendre la voix).
        if _NUMBER_RE.search(nu) or _LISTE_NUM_RE.search(nu) or bas.endswith(_ABBREV):
            continue

        phrases.append(candidat.strip())
        pos = end

    reste = buffer[pos:]

    phrases = _hard_wrap(phrases, max_chars, min_chars)

    # Un « reste » sans ponctuation ne doit pas grossir sans fin : sinon un modèle qui
    # enchaîne des centaines de caractères sans point ne déclencherait jamais la voix.
    # On émet donc un morceau dès que le reste dépasse la limite — sauf s'il contient un
    # bloc de code ouvert, qu'on garde intact en attendant sa fermeture.
    if "```" not in reste:
        while len(reste) > max_chars:
            cut = reste.rfind(" ", 0, max_chars)
            if cut <= 0:
                break  # un seul « mot » démesuré : on attend la suite du flux
            morceau = reste[:cut].strip()
            if len(morceau) >= min_chars:
                phrases.append(morceau)
            reste = reste[cut:]

    return phrases, reste


def _hard_wrap(items: list[str], max_chars: int, min_chars: int) -> list[str]:
    """Coupe les segments trop longs et jette les miettes insignifiantes."""
    out: list[str] = []
    for item in items:
        while len(item) > max_chars:
            cut = item.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            out.append(item[:cut].strip())
            item = item[cut:].strip()
        if len(item) >= min_chars:
            out.append(item)
    return [p for p in out if p]


# Emoji et pictogrammes : Kokoro n'en fait rien d'utile (au mieux il les ignore,
# au pire il les épelle). On les retire avant la synthèse.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001F2FF"  # symboles et pictogrammes
    "\U0001F300-\U0001FAFF"  # émoticônes et compléments
    "\U00002600-\U000027BF"  # symboles divers et dingbats
    "\u2B00-\u2BFF"          # flèches et symboles supplémentaires
    "\uFE0F"                 # sélecteur de variante
    "]+"
)

# Règles horizontales markdown : ---, ***, ___, ===
_REGLE_RE = re.compile(r"^[-*_=]{3,}$")

# Cellule de séparation d'un tableau markdown : « --- », « :--: », …
_TABLE_SEP_RE = re.compile(r"^:?-{2,}:?$")

# Abréviations françaises et unités, remplacées par ce qu'il faut prononcer.
# Les sigles demandent une frontière de mot stricte, sinon « Pr » mangerait
# « Premier » et « env. » « enveloppe ».
_ABREVIATIONS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bM\.\s+(?=[A-ZÉÈÀÂÎÔÛ])"), "Monsieur "),
    (re.compile(r"\bMM\.\s+(?=[A-ZÉÈÀÂÎÔÛ])"), "Messieurs "),
    (re.compile(r"\bMme\b"), "Madame"),
    (re.compile(r"\bMlle\b"), "Mademoiselle"),
    (re.compile(r"\bDr\b"), "Docteur"),
    (re.compile(r"\bPr\b"), "Professeur"),
    (re.compile(r"\bn°\s*"), "numéro "),
    (re.compile(r"\bcf\.\s*"), "voir "),
    (re.compile(r"\bp\.\s?ex\.\s*"), "par exemple "),
    (re.compile(r"\betc\.(?=\s|$)"), "et cetera"),
    (re.compile(r"\bav\.\s+(?=[A-ZÉÈÀ])"), "avenue "),
    (re.compile(r"\bbd\.\s+(?=[A-ZÉÈÀ])"), "boulevard "),
    (re.compile(r"\bkm/h\b"), "kilomètres par heure"),
    (re.compile(r"(\d)\s*%"), r"\1 pour cent"),
    (re.compile(r"(\d)\s*°\s*C\b"), r"\1 degrés"),
    (re.compile(r"(\d)\s*€"), r"\1 euros"),
    (re.compile(r"\s&amp;\s|\s&\s"), " et "),
    (re.compile(r"\s*→\s*"), " puis "),
)


def _nettoyer_ligne(ligne: str) -> str:
    """Retire d'une ligne le balisage qui serait lu comme du texte."""
    l = ligne.strip()
    if not l or _REGLE_RE.fullmatch(l):
        return ""
    if l.startswith("|") and l.endswith("|"):  # ligne de tableau
        cellules = [cellule.strip() for cellule in l.strip("|").split("|")]
        if all(not c or _TABLE_SEP_RE.fullmatch(c) for c in cellules):
            return ""  # « |---|---| » n'est pas du texte
        # On lit une ligne de tableau cellule par cellule : « Réglage : Gain »
        # s'entend mieux que « RéglageGain », et que les barres verticales.
        return ", ".join(c for c in cellules if c)
    l = re.sub(r"^#{1,6}\s*", "", l)  # titre
    l = re.sub(r"^>\s*", "", l)  # citation
    l = re.sub(r"^[-*•–—]\s+", "", l)  # puce
    l = re.sub(r"^\d+[.)]\s+", "", l)  # liste numérotée : « 1. » ne se lit pas
    return l


def clean_for_speech(text: str) -> str:
    """Prépare un fragment pour la synthèse : rien ne doit être lu comme du balisage.

    Appelée **par phrase** (juste avant la synthèse), donc chaque fragment peut
    arriver tronqué : les traitements sont tous indépendants et sans état.
    """
    # 1. Blocs de code et code inline.
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # 2. Liens et images. L'ordre compte : sur « ![alt](url) » il ne faut pas
    #    laisser le « alt » devenir du texte, alors que sur « [texte](url) » le
    #    « texte » est justement ce qu'on veut lire. Un ancien code retirait
    #    l'URL seule et laissait « [texte]( » — l'auditeur entendait les crochets.
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)
    text = re.sub(r"https?://\S+", "lien", text)

    # 3. Structure de ligne (titres, puces, tableaux, règles).
    text = " ".join(f for f in (_nettoyer_ligne(l) for l in text.splitlines()) if f)

    # 4. Emphase restante, emoji, abréviations et unités.
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<=\s)\*(\S[^*]*?)\*(?=\s|$)", r"\1", text)
    text = re.sub(r"[*_#>`]", "", text)
    text = _EMOJI_RE.sub(" ", text)
    # Milliers « 1 000 » (espace normale, insécable ou fine) → « 1000 » : sinon le
    # synthétiseur énumère les chiffres un par un.
    text = re.sub(r"(\d)[\u00a0\u202f\s](?=\d{3}\b)", r"\1", text)
    for motif, remplacement in _ABREVIATIONS:
        text = motif.sub(remplacement, text)

    return re.sub(r"\s+", " ", text).strip()
