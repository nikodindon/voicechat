"""Persistance des conversations : /save, /load, /conversations, reprise au lancement.

Un fichier JSON par conversation, dans un dossier de données XDG. Rien n'est chargé
automatiquement : c'est l'appelant qui décide (option ``--continue`` ou commande).
"""

from __future__ import annotations

import json
import os
import re
import time
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
    fichier = nom_fichier(nom)
    dossier_cible = dossier_profils()
    for extension in (".md", ".txt"):
        chemin = dossier_cible / f"{fichier}{extension}"
        if chemin.is_file():
            return chemin.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"aucun profil « {nom} » dans {dossier_cible}")


def enregistrer_profil(nom: str, prompt: str) -> Path:
    """Écrit un profil de prompt système."""
    if not prompt.strip():
        raise ValueError("prompt vide")
    chemin = dossier_profils() / f"{nom_fichier(nom)}.md"
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(prompt.strip() + "\n", encoding="utf-8")
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
