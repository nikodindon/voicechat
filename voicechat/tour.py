"""Déroulé d'un tour : d'un flux de tokens vers du texte et des phrases.

Ce module existe pour **une seule raison** : la console et le client web doivent découper
les phrases exactement de la même façon. Si les deux faisaient leur propre boucle, l'une
finirait par recevoir une correction que l'autre n'aurait pas — et l'écran et la voix
divergeraient sans que rien ne le signale. C'est précisément le genre d'écart silencieux
qu'on cherche à éviter ici ; la découpe des phrases est d'ailleurs la partie la plus
retouchée du projet (v0.5, v1.1).

Le découpage reste « en flux » : un fragment sans ponctuation finale est gardé pour le
morceau suivant, sinon on entendrait des bouts de phrase.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from .text import split_sentences


def derouler(
    flux: Iterator[str],
    on_texte: Callable[[str], None] | None = None,
    on_phrase: Callable[[str], None] | None = None,
    sur_morceau: Callable[[], None] | None = None,
    arret: Callable[[], bool] | None = None,
    decoupeur: Callable[[str], tuple[list[str], str]] | None = None,
) -> tuple[str, str, bool]:
    """Consomme un flux de tokens, en tire le texte et les phrases complètes.

    * ``on_texte``   : appelé pour chaque morceau reçu (l'affichage, l'envoi SSE…) ;
    * ``on_phrase``  : appelé pour chaque phrase **complète** — c'est ce qui part à la voix ;
    * ``sur_morceau``: effet de bord après chaque morceau, qui **n'arrête rien**. C'est là
      que la console surveille son clavier : Échap coupe la voix mais la génération
      continue, et surtout la réponse reste entière ;
    * ``arret``      : consulté après chaque morceau ; s'il rend ``True``, on sort de la
      boucle (le client web a fermé l'onglet).

    Les deux sont distincts à dessein : les confondre faisait qu'Échap interrompait la
    lecture du flux, donc tronquait la réponse — ce que la vérification d'interruption
    a attrapé tout de suite.

    Retourne ``(reponse, reste, arrete)`` :

    * ``reponse`` : tout le texte reçu, tel qu'il a été affiché ;
    * ``reste``   : le dernier fragment, sans ponctuation finale. Il n'a pas été passé à
      ``on_phrase`` **exprès** : il faut le dire ou l'envoyer quand même, après la boucle,
      car c'est la fin de la réponse ;
    * ``arrete``  : ``True`` si l'arrêt a été demandé.
    """
    decouper = decoupeur or split_sentences
    reponse = ""
    tampon = ""
    for morceau in flux:
        reponse += morceau
        tampon += morceau
        if on_texte is not None:
            on_texte(morceau)
        phrases, tampon = decouper(tampon)
        for phrase in phrases:
            if on_phrase is not None:
                on_phrase(phrase)
        if sur_morceau is not None:
            sur_morceau()
        if arret is not None and arret():
            return reponse, tampon, True
    return reponse, tampon, False
