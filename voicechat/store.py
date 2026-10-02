"""Persistance des conversations : /save, /load, /conversations, reprise au lancement.

Un fichier JSON par conversation, dans un dossier de données XDG. Rien n'est chargé
automatiquement : c'est l'appelant qui décide (option ``--continue`` ou commande).
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 1
NOM_DEFAUT = "derniere"


def dossier() -> Path:
    """Dossier des conversations : ``$VOICECHAT_DATA``, sinon XDG_DATA_HOME, sinon ~/.local/share."""
    force = os.environ.get("VOICECHAT_DATA")
    if force:
        racine = Path(force).expanduser()
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        racine = base / "voicechat"
    return racine / "conversations"


def nom_fichier(nom: str) -> str:
    """Nettoie un nom de conversation pour en faire un nom de fichier sûr.

    Les espaces et accents deviennent des tirets ; les séparateurs de chemin sont
    neutralisés, donc ``/save ../../etc/passwd`` ne peut pas sortir du dossier.
    """
    propre = re.sub(r"[^A-Za-z0-9._-]+", "-", nom.strip()).strip("-.")
    if not propre:
        raise ValueError("nom de conversation vide ou invalide")
    return propre[:80]


@dataclass
class Conversation:
    """Une conversation sauvegardée, telle qu'échangée avec l'API."""

    nom: str
    messages: list[dict] = field(default_factory=list)
    modele: str = ""
    systeme: str = ""
    cree: float = field(default_factory=time.time)
    maj: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "nom": self.nom,
            "modele": self.modele,
            "systeme": self.systeme,
            "cree": self.cree,
            "maj": self.maj,
            "messages": self.messages,
        }

    @property
    def echanges(self) -> int:
        """Nombre de messages hors message système."""
        return sum(1 for m in self.messages if m.get("role") != "system")

    @classmethod
    def from_dict(cls, data: dict, nom: str = "") -> "Conversation":
        if not isinstance(data, dict):
            raise ValueError("le fichier ne contient pas un objet JSON")
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ValueError("champ « messages » absent ou invalide")
        propre: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            contenu = message.get("content")
            if role in ("system", "user", "assistant") and isinstance(contenu, str):
                propre.append({"role": role, "content": contenu})
        return cls(
            nom=data.get("nom") or nom,
            messages=propre,
            modele=data.get("modele") or "",
            systeme=data.get("systeme") or "",
            cree=float(data.get("cree") or time.time()),
            maj=float(data.get("maj") or time.time()),
        )


# ------------------------------------------------------------------- écriture
def enregistrer(
    nom: str,
    messages: list[dict],
    *,
    modele: str = "",
    systeme: str = "",
    cree: float | None = None,
) -> Path:
    """Écrit la conversation sur disque et retourne le chemin du fichier."""
    fichier = nom_fichier(nom)
    dossier_cible = dossier()
    dossier_cible.mkdir(parents=True, exist_ok=True)
    chemin = dossier_cible / f"{fichier}.json"

    conversation = Conversation(
        nom=fichier,
        messages=list(messages),
        modele=modele,
        systeme=systeme,
        cree=cree if cree is not None else time.time(),
    )
    # Écriture atomique : un fichier temporaire puis un remplacement, pour ne jamais
    # laisser une conversation à moitié écrite si le processus est interrompu.
    temporaire = chemin.with_suffix(".json.tmp")
    temporaire.write_text(
        json.dumps(conversation.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporaire.replace(chemin)
    return chemin


# -------------------------------------------------------------------- lecture
def charger(nom: str) -> Conversation:
    """Charge une conversation. Lève FileNotFoundError ou ValueError, jamais autre chose."""
    chemin = dossier() / f"{nom_fichier(nom)}.json"
    try:
        texte = chemin.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"aucune conversation nommée « {nom} »") from None
    try:
        data = json.loads(texte)
    except json.JSONDecodeError as exc:
        raise ValueError(f"fichier illisible ({exc})") from exc
    return Conversation.from_dict(data, nom=chemin.stem)


def lister() -> list[dict]:
    """Résumé des conversations sauvegardées, la plus récente d'abord."""
    dossier_cible = dossier()
    if not dossier_cible.is_dir():
        return []

    resume: list[dict] = []
    for chemin in dossier_cible.glob("*.json"):
        try:
            conversation = charger(chemin.stem)
        except (ValueError, OSError):
            continue  # fichier corrompu : on l'ignore plutôt que de tout casser
        resume.append(
            {
                "nom": conversation.nom,
                "echanges": conversation.echanges,
                "maj": conversation.maj,
                "modele": conversation.modele,
            }
        )
    return sorted(resume, key=lambda c: c["maj"], reverse=True)


def derniere() -> Conversation | None:
    """La conversation la plus récente, ou None s'il n'y en a aucune."""
    sauvegardes = lister()
    if not sauvegardes:
        return None
    try:
        return charger(sauvegardes[0]["nom"])
    except (FileNotFoundError, ValueError):
        return None


def supprimer(nom: str) -> bool:
    """Supprime une conversation. Retourne True si un fichier a été enlevé."""
    chemin = dossier() / f"{nom_fichier(nom)}.json"
    try:
        chemin.unlink()
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------- recherche
@dataclass
class Trouvaille:
    """Un passage retrouvé dans une conversation sauvegardée."""

    conversation: str
    role: str
    message: int
    date: float
    extrait: str

    def role_lisible(self) -> str:
        """Le rôle tel qu'on l'affiche dans la console."""
        return {"user": "vous", "assistant": "ia", "system": "système"}.get(self.role, self.role)


def sans_accents(texte: str) -> str:
    """Minuscules sans accents : « Résumé » et « resume » doivent se retrouver.

    Chercher « resume » sans trouver « résumé » serait une source d'échecs silencieux
    particulièrement pénible en français. NFKD sépare la lettre de son accent, puis on
    retire les diacritiques. La longueur est préservée pour les lettres accentuées
    («é» → «e»), mais pas pour les ligatures («œ» → «oe») : d'où la borne dans
    ``chercher()``.
    """
    decompose = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _extrait(contenu: str, position: int, largeur: int = 45) -> str:
    """Le passage autour de la correspondance, encadré de points de suspension."""
    debut = max(0, position - largeur)
    fin = min(len(contenu), position + largeur)
    morceau = " ".join(contenu[debut:fin].split())  # sauts de ligne et doubles espaces
    return ("… " if debut else "") + morceau + (" …" if fin < len(contenu) else "")


def chercher(motif: str, limite: int = 40) -> list[Trouvaille]:
    """Cherche un motif dans toutes les conversations sauvegardées.

    Les fichiers sont relus depuis le disque (ils sont petits) : pas d'index à tenir à
    jour, donc rien qui puisse devenir faux en silence. Résultat : les conversations les
    plus récentes d'abord, au plus ``limite`` passages.
    """
    aiguille = sans_accents(motif.strip())
    if not aiguille:
        return []

    trouvailles: list[Trouvaille] = []
    for resume in lister():  # déjà trié, la plus récente d'abord
        try:
            conversation = charger(resume["nom"])
        except (FileNotFoundError, ValueError, OSError):
            continue  # conversation abîmée : on l'ignore au lieu d'abandonner la recherche
        for index, message in enumerate(conversation.messages):
            contenu = message.get("content") or ""
            position = sans_accents(contenu).find(aiguille)
            if position < 0:
                continue
            trouvailles.append(
                Trouvaille(
                    conversation=conversation.nom,
                    role=message.get("role", "?"),
                    message=index,
                    date=conversation.maj,
                    extrait=_extrait(contenu, min(position, max(0, len(contenu) - 1))),
                )
            )
            if len(trouvailles) >= limite:
                return trouvailles
    return trouvailles


# --------------------------------------------------------------------- profils
def dossier_profils() -> Path:
    """Profils de prompt système : ``$VOICECHAT_PROFILS``, sinon XDG_CONFIG_HOME."""
    force = os.environ.get("VOICECHAT_PROFILS")
    if force:
        return Path(force).expanduser()
    config = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config).expanduser() if config else Path.home() / ".config"
    return base / "voicechat" / "profils"


def lister_profils() -> list[str]:
    """Noms des profils disponibles (fichiers .md ou .txt du dossier des profils)."""
    dossier_cible = dossier_profils()
    if not dossier_cible.is_dir():
        return []
    trouves = {p.stem for p in dossier_cible.iterdir() if p.suffix in (".md", ".txt")}
    return sorted(trouves)


def charger_profil(nom: str) -> str:
    """Contenu d'un profil de prompt. Lève FileNotFoundError s'il n'existe pas."""
    return lire_profil(nom).prompt


# Clés reconnues en en-tête d'un profil. Une clé inconnue n'est pas consommée :
# elle fait partie du prompt, pour ne jamais perdre de texte par inadvertance.
_CLES_PROFIL = ("voix", "vitesse", "langue")
_ENTETE_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.*)$")


@dataclass
class Profil:
    """Un profil de persona : le prompt système, plus des réglages optionnels.

    L'en-tête est facultatif et se place au tout début du fichier ::

        voix: ff_siwis:4+ef_dora:1
        vitesse: 1.1

        Tu es un assistant...
    """

    prompt: str
    voix: str | None = None
    vitesse: float | None = None
    langue: str | None = None


def analyser_profil(texte: str) -> Profil:
    """Sépare l'en-tête de réglages du prompt. Fonction pure."""
    voix = vitesse = langue = None
    lignes = texte.splitlines()
    i = 0
    while i < len(lignes):
        ligne = lignes[i]
        if not ligne.strip():
            break  # ligne vide = fin de l'en-tête
        correspondance = _ENTETE_RE.match(ligne)
        if not correspondance or correspondance.group(1).lower() not in _CLES_PROFIL:
            break  # ce n'est pas un réglage : le prompt commence ici
        cle = correspondance.group(1).lower()
        valeur = correspondance.group(2).strip()
        if cle == "voix":
            voix = valeur or None
        elif cle == "langue":
            langue = valeur or None
        elif cle == "vitesse":
            try:
                vitesse = float(valeur)
            except ValueError:
                vitesse = None  # valeur illisible : on l'ignore plutôt que planter
        i += 1
    # La ligne vide qui sépare l'en-tête du prompt est sautée, mais pas celles
    # qui suivent (elles appartiennent au prompt).
    if i < len(lignes) and not lignes[i].strip():
        i += 1
    return Profil(
        prompt="\n".join(lignes[i:]).strip(),
        voix=voix,
        vitesse=vitesse,
        langue=langue,
    )


def lire_profil(nom: str) -> Profil:
    """Charge un profil avec ses réglages. Lève FileNotFoundError s'il n'existe pas."""
    fichier = nom_fichier(nom)
    dossier_cible = dossier_profils()
    for extension in (".md", ".txt"):
        chemin = dossier_cible / f"{fichier}{extension}"
        if chemin.is_file():
            return analyser_profil(chemin.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"aucun profil « {nom} » dans {dossier_cible}")


def enregistrer_profil(
    nom: str,
    prompt: str,
    voix: str | None = None,
    vitesse: float | None = None,
    langue: str | None = None,
) -> Path:
    """Écrit un profil de prompt système, avec ses réglages en en-tête.

    Les réglages ne sont écrits que s'ils sont fournis : un profil sans voix reste
    un simple fichier de prompt, comme avant.
    """
    if not prompt.strip():
        raise ValueError("prompt vide")
    entete = ""
    if langue:
        entete += f"langue: {langue}\n"
    if voix:
        entete += f"voix: {voix}\n"
    if vitesse:
        entete += f"vitesse: {vitesse}\n"
    if entete:
        entete += "\n"
    chemin = dossier_profils() / f"{nom_fichier(nom)}.md"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(entete + prompt.strip() + "\n", encoding="utf-8")
    return chemin


# ---------------------------------------------------------------------- export
def exporter_markdown(
    messages: list[dict],
    chemin: Path,
    *,
    titre: str = "Conversation",
    modele: str = "",
    voix: str = "",
) -> Path:
    """Écrit la conversation en markdown lisible et retourne le chemin du fichier."""
    lignes = [f"# {titre}", ""]
    entetes = [f"- **Date** : {time.strftime('%d/%m/%Y %H:%M', time.localtime())}"]
    if modele:
        entetes.append(f"- **Modèle** : {modele}")
    if voix:
        entetes.append(f"- **Voix** : {voix}")
    lignes += entetes + [""]

    systeme = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    if systeme:
        lignes += ["## Prompt système", "", systeme, "", "---", ""]

    lignes += ["## Échanges", ""]
    for message in messages:
        role = message.get("role")
        contenu = (message.get("content") or "").strip()
        if not contenu or role not in ("user", "assistant"):
            continue
        lignes += ["**Vous**" if role == "user" else "**Assistant**", "", contenu, ""]

    chemin = Path(chemin).expanduser()
    if chemin.parent != Path(""):
        chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text("\n".join(lignes).rstrip() + "\n", encoding="utf-8")
    return chemin
