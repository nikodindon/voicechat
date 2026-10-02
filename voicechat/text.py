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

    for match in _SENT_RE.finditer(buffer):
        end = match.end()
        candidat = buffer[pos:end]
        if not candidat.strip():
            pos = end
            continue

        nu = candidat.rstrip()
        bas = nu.lower()
        # Abréviation connue suivie d'autre texte → ce n'est pas une frontière.
        if end < len(buffer) and (bas.endswith(_ABBREV) or _NUMBER_RE.search(nu)):
            continue

        phrases.append(candidat.strip())
        pos = end

    reste = buffer[pos:]

    phrases = _hard_wrap(phrases, max_chars, min_chars)

    # Un « reste » sans ponctuation ne doit pas grossir sans fin : sinon un modèle qui
    # enchaîne des centaines de caractères sans point ne déclencherait jamais la voix.
    # On émet donc un morceau dès que le reste dépasse la limite.
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


def clean_for_speech(text: str) -> str:
    """Retire ce qui sonne mal à l'oral (code, liens, markdown)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)  # blocs de code
    text = re.sub(r"`([^`]*)`", r"\1", text)  # code inline
    text = re.sub(r"https?://\S+", "lien", text)
    text = re.sub(r"\*\*|__|[*_#>]", "", text)  # markdown
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)  # puces
    return re.sub(r"\s+", " ", text).strip()
