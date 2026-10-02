"""Mode dialogue : deux personas qui se répondent, chacun avec sa voix (v1.5).

Tout existait déjà — les profils (prompt + voix + vitesse + langue + modèle), la voix
portée par le profil, le mélange de voix pondéré, la réserve d'audio. Il ne manquait que
l'alternance.

**La ligne qui porte le mode** est d'appliquer le profil du persona avant son tour
(``preparer_voix``) : c'est elle qui fait qu'Alice parle avec la voix d'Alice.

Le reste n'est que de la mise en scène : chaque persona a **sa** liste de messages, où
l'autre est « l'utilisateur ». C'est ce qui permet de réutiliser tel quel le chemin normal
d'un tour — le modèle voit une conversation ordinaire, où il tient toujours le même rôle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from . import store

if TYPE_CHECKING:  # évite un cycle : cli.py importe ce module
    from .cli import Tour

TOURS_DEFAUT = 6
# Au-delà, une réplique n'est plus une réplique mais un laïus : on le signale. 35 mots
# tiennent en une quinzaine de secondes de voix, ce qui est déjà long dans un échange.
MOTS_MAX = 90


class SessionDialogue(Protocol):
    """Ce que le mode dialogue attend d'une session.

    Trois choses empruntées à la console — la configuration, le synthétiseur, la file de
    la voix — et deux gestes : appliquer les réglages d'un profil, jouer un tour de
    modèle. Le contrat permet aux tests d'utiliser une doublure sans la faire passer pour
    un `ChatSession` par un `type: ignore`.
    """

    cfg: Any
    tts: Any
    speech: Any

    def appliquer_reglages(self, profil: store.Profil) -> None: ...

    def tour_modele(
        self, messages: list[dict], etiquette: str = ..., modele: str | None = ...
    ) -> "Tour": ...

# Le tout premier message de la discussion. Ensuite, chacun reçoit ce que l'autre vient de
# dire, sans décoration : **qui est en face** est dit dans le prompt système (voir
# `_preparer_les_prompts`), pas dans le texte du message. Deux essais ont montré pourquoi :
#
# * sans rien, Bob lisait la réplique d'Alice sans savoir qu'elle venait d'Alice et
#   répondait « Exactement, Bob » — en s'appelant lui-même ;
# * en préfixant chaque réplique par « alice : », le modèle a **imité le format** et
#   commencé ses propres réponses par « bob : » — préfixe qui partait ensuite à la voix.
#
# Le nom appartient donc aux prompts, et le texte reste du texte.
CONSIGNE = "Sujet : {sujet}\n\nOuvre la discussion en deux ou trois phrases."
# La règle qui empêche les deux dérives observées à l'usage : le modèle écrit « bob : » en
# début de réponse (le nom part alors à la voix : « bob, deux points, … »), et il recopie
# le message reçu avant d'y répondre. Interdire explicitement est plus sûr que nettoyer
# après coup — quand la phrase fautive arrive, elle est déjà partie à la synthèse.
EN_FACE = (
    "\n\nTu discutes avec {autre}. Tu lui réponds directement, sans écrire « {nom} » ni "
    "« {autre} » en début de réponse, et sans recopier ce qu'il vient de te dire."
)


@dataclass
class Persona:
    """Un des interlocuteurs : son nom, son profil, et sa propre histoire.

    ``messages`` est une conversation complète et séparée, avec le prompt du persona en
    tête. L'autre persona y apparaît comme « l'utilisateur » : vu du modèle, c'est une
    discussion normale où il tient toujours le même rôle — bien plus facile à suivre pour
    lui qu'un historique unique où les rôles s'inverseraient à chaque tour.
    """

    nom: str
    profil: store.Profil
    messages: list[dict] = field(default_factory=list)

    @property
    def voix(self) -> str:
        """La voix demandée par le profil (vide s'il n'en impose pas)."""
        return self.profil.voix or ""

    @classmethod
    def charger(cls, nom: str) -> "Persona":
        """Charge un persona depuis un profil enregistré (``/profil save``).

        Lève ``FileNotFoundError`` si le profil n'existe pas, ``ValueError`` s'il ne
        contient aucun prompt.
        """
        profil = store.lire_profil(nom)
        if not profil.prompt.strip():
            raise ValueError(f"le profil « {nom} » ne contient aucun prompt")
        return cls(
            nom=store.nom_fichier(nom),
            profil=profil,
            messages=[{"role": "system", "content": profil.prompt}],
        )


class Dialogue:
    """Fait dialoguer deux personas, chacun avec sa voix, sur un sujet.

    Volontairement simple : le premier ouvre sur le sujet, le second lui répond, le
    premier répond à la réponse, et ainsi de suite. Aucun des deux ne décide de s'arrêter
    — c'est le nombre de répliques qui décide, sinon une conversation entre deux modèles
    est sans fin par construction.
    """

    def __init__(
        self,
        session: SessionDialogue,
        a: Persona,
        b: Persona,
        sujet: str,
        tours: int = TOURS_DEFAUT,
    ) -> None:
        self.session = session
        self.personas = (a, b)
        self.sujet = sujet.strip()
        self.tours = max(2, tours)
        self.mots_max = MOTS_MAX
        # Les répliques produites : (nom du persona, texte). Rendu par `derouler`, ce qui
        # permet aux tests et à la vérification de lire le dialogue sans dépiauger la
        # sortie de la console.
        self.repliques: list[tuple[str, str]] = []

    # ------------------------------------------------------------------- voix
    def voix_effectives(self) -> tuple[str, str]:
        """Les voix qui vont réellement parler, dans l'ordre.

        Un profil sans ``voix:`` ne change rien : le persona parlerait donc avec la voix
        restée en place — celle de l'autre. C'est précisément ce qu'un dialogue à deux
        voix ne doit pas faire, et ça se voit dans ce calcul avant de commencer.
        """
        defaut = self.session.cfg.voice
        return tuple(p.voix or defaut for p in self.personas)  # type: ignore[return-value]

    def annoncer(self) -> None:
        """Dit qui parle, avec quelle voix, et prévient si les deux se ressemblent."""
        a, b = self.personas
        voix_a, voix_b = self.voix_effectives()
        print(f"dialogue : {a.nom} ↔ {b.nom} — {self.tours} réplique(s)")
        print(f"  {a.nom} : voix {voix_a or '(défaut)'}"
              + (f" · modèle {a.profil.modele}" if a.profil.modele else ""))
        print(f"  {b.nom} : voix {voix_b or '(défaut)'}"
              + (f" · modèle {b.profil.modele}" if b.profil.modele else ""))
        if not voix_a or voix_a == voix_b:
            print("  [attention] les deux personas ont la même voix : le dialogue")
            print("             s'entendra, mais on ne saura pas qui parle.")
            print("             Ajouter « voix: <nom> » en tête du profil de l'un des deux.")

    def preparer_voix(self, persona: Persona) -> None:
        """Met la voix du persona en place, après avoir vidé la file de synthèse.

        ``SpeechPipeline`` synthétise dans un thread et lit ``tts.voice`` au **dépilement**,
        pas à l'empilement : basculer avec des phrases encore en attente ferait dire à cette
        persona les répliques de la précédente.

        **Mesuré, pourtant : ce vidage n'est pas ce qui nous sauve.** ``tour_modele`` attend
        déjà la fin de la voix à la fin de chaque tour (``_attendre_voix``), donc la file
        est vide quand on arrive ici — et retirer cette attente ne fait rien échouer dans la
        vérification (essayé). C'est une assurance, pas la pièce maîtresse : elle couvre les
        chemins où un tour se termine sans attendre (voix coupée, interruption, panne), et
        elle ne coûte rien quand la file est vide.

        La ligne qui compte vraiment est ``appliquer_reglages``, juste en dessous : c'est
        elle qui fait qu'Alice parle avec la voix d'Alice.
        """
        session = self.session
        if session.speech is not None:
            session.speech.wait()
        if session.tts is None:
            return
        session.appliquer_reglages(persona.profil)

    # ---------------------------------------------------------------- déroulé
    def _preparer_les_prompts(self) -> None:
        """Dit à chacun qui est en face — dans le prompt, pas dans le message.

        Voir l'en-tête du module : mettre le nom dans le texte de la réplique fait imiter
        le format par le modèle, qui se met à répondre « bob : … ». Le prompt système, lui,
        donne l'information sans fournir de format à copier.
        """
        a, b = self.personas
        for personne, autre in ((a, b), (b, a)):
            personne.messages[0] = {
                "role": "system",
                "content": personne.profil.prompt
                + EN_FACE.format(autre=autre.nom, nom=personne.nom),
            }

    def derouler(self) -> list[tuple[str, str]]:
        """Joue le dialogue et rend les répliques produites."""
        self._preparer_les_prompts()
        entrant = CONSIGNE.format(sujet=self.sujet)
        etat = self._etat_voix()
        try:
            for i in range(self.tours):
                persona = self.personas[i % 2]
                tour = self._tour(persona, entrant)
                if tour.interrompu:
                    print("[Ctrl+C] dialogue interrompu")
                    break
                if not tour.reponse.strip():
                    # Panne réseau ou réponse vide : insister ne servirait à rien, le
                    # suivant n'aurait rien à répondre.
                    break
                self.repliques.append((persona.nom, tour.reponse))
                entrant = tour.reponse
        finally:
            self._restaurer_voix(etat)
        return self.repliques

    def _tour(self, persona: Persona, entrant: str):
        """Une réplique : sa voix, puis un tour de modèle ordinaire."""
        self.preparer_voix(persona)
        persona.messages.append({"role": "user", "content": entrant})
        tour = self.session.tour_modele(
            persona.messages,
            f"{persona.nom} › ",
            persona.profil.modele,
        )
        self._avertir_si_bavard(persona, tour.reponse)
        return tour

    def _avertir_si_bavard(self, persona: Persona, texte: str) -> None:
        """Prévient quand une réplique part en pavé.

        Un dialogue s'écoute : une réponse de 200 mots avec des titres et des listes dure
        deux minutes et se lit comme un rapport. Le modèle ne peut pas le savoir — c'est au
        persona de le dire (voir les exemples du README), et à nous de le signaler.
        """
        mots = len(texte.split())
        if mots > self.mots_max:
            print(
                f"  [dialogue] {persona.nom} : réponse de {mots} mots — longue à écouter. "
                f"Ajouter « au maximum {self.mots_max} mots, pas de listes » à son profil."
            )

    # -------------------------------------------------------------- la voix après
    def _etat_voix(self) -> tuple[str, float, str]:
        """La voix de la session avant le dialogue."""
        cfg = self.session.cfg
        return cfg.voice, cfg.speed, cfg.lang

    def _restaurer_voix(self, etat: tuple[str, float, str]) -> None:
        """Rend à la session la voix qu'elle avait.

        Sans ça, la console resterait avec la voix du dernier persona : on lui parle, et
        c'est un des deux personnages qui répond. Dérouterait pour rien.
        """
        voix, vitesse, langue = etat
        cfg = self.session.cfg
        tts = self.session.tts
        if tts is not None and self.session.speech is not None:
            self.session.speech.wait()
        if tts is not None:
            if langue and langue != cfg.lang:
                tts.set_lang(langue)
            if voix and voix != tts.voice:
                tts.set_voice(voix)
            tts.speed = vitesse
        cfg.voice, cfg.speed, cfg.lang = voix, vitesse, langue
