"""Configuration : variables d'environnement + ligne de commande fusionnées."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

DEFAULT_BASE_URL = "http://100.91.114.49:8080/v1"
DEFAULT_VOICE = "ff_siwis"  # français féminin
DEFAULT_LANG = "f"
DEFAULT_SPEED = 1.0
DEFAULT_SYSTEM = (
    "Tu es un assistant francophone concis et précis. "
    "Réponds en phrases courtes et parlées, sans listes à puces ni markdown."
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def nom_court(modele: str) -> str:
    """Nom lisible d'un modèle, pour l'affichage seulement.

    llama.cpp renvoie comme identifiant un chemin complet
    (« /mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf »). On n'en garde que
    le nom de fichier sans extension. Ce nom n'est **pas** envoyé au serveur : le vrai
    identifiant reste ``cfg.model``.
    """
    if not modele:
        return "?"
    return Path(modele).name.removesuffix(".gguf") or modele


def _env_bool(name: str, default: bool = True) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw not in ("0", "false", "no", "off", "")


def _liste_urls(brut: str) -> list[str]:
    """« url1, url2 » → liste de serveurs, dans l'ordre écrit.

    Accepte la virgule, le point-virgule et le saut de ligne comme séparateurs :
    une longue liste de secours se relit mieux sur plusieurs lignes dans un .env.
    """
    if isinstance(brut, (list, tuple)):  # déjà une liste (fichier TOML)
        return [str(u).strip().rstrip("/") for u in brut if str(u).strip()]
    propres = (brut or "").replace(";", ",").replace("\n", ",")
    return [u.strip().rstrip("/") for u in propres.split(",") if u.strip()]


# --------------------------------------------------------------- fichier TOML
def dossier_config() -> Path:
    """``$XDG_CONFIG_HOME/voicechat``, sinon ``~/.config/voicechat``."""
    base = os.environ.get("XDG_CONFIG_HOME")
    racine = Path(base).expanduser() if base else Path.home() / ".config"
    return racine / "voicechat"


def chemin_config() -> Path:
    """Le fichier de configuration TOML : ``$VOICECHAT_CONFIG`` ou l'emplacement XDG."""
    force = os.environ.get("VOICECHAT_CONFIG")
    return Path(force).expanduser() if force else dossier_config() / "config.toml"


def lire_config(chemin: Path | None = None) -> tuple[dict, str | None]:
    """Lit le fichier TOML. Renvoie ``(reglages, avertissement)``.

    Ne lève jamais : un fichier de configuration abîmé ne doit pas empêcher de
    démarrer. Le fichier est signalé à l'appelant, qui décide quoi en dire.
    """
    fichier = chemin or chemin_config()
    if not fichier.is_file():
        return {}, None
    try:
        import tomllib

        with fichier.open("rb") as flux:
            donnees = tomllib.load(flux)
    except ImportError:  # Python < 3.11 : tomllib est apparu là
        return {}, (
            f"{fichier} ignoré : la lecture TOML demande Python 3.11+ "
            f"(tomllib est dans la bibliothèque standard à partir de là)"
        )
    except Exception as exc:
        return {}, f"{fichier} illisible ({exc}) — on continue sans lui"
    section = donnees.get("voicechat")
    if section is None:
        return {}, f"{fichier} ne contient pas de section [voicechat]"
    if not isinstance(section, dict):
        return {}, f"{fichier} : [voicechat] doit être une table"

    # On ne garde que les clés qui correspondent à un vrai réglage (la liste vient de la
    # définition de Config, pour que les deux ne puissent pas diverger). Une clé inconnue
    # est signalée : une faute de frappe silencieuse serait le pire des cas — le réglage
    # ne s'appliquerait jamais sans qu'on sache pourquoi.
    connues = {champ.name for champ in fields(Config)}
    reglages = {c: v for c, v in section.items() if c in connues}
    inconnues = sorted(set(section) - connues)

    # Les réglages par modèle vivent au **premier niveau** du fichier, pas dans
    # [voicechat] — c'est la forme naturelle à écrire :
    #   [modeles.gros-modele]
    #   voice = "..."
    modeles = donnees.get("modeles")
    if isinstance(modeles, dict):
        reglages["modeles"] = modeles

    hors = sorted(set(donnees) - {"voicechat", "modeles"})
    soucis = []
    if inconnues:
        soucis.append(f"clé(s) inconnue(s) dans [voicechat] : {', '.join(inconnues)}")
    if hors:
        soucis.append(f"section(s) inconnue(s) : {', '.join(hors)}")
    return reglages, (f"{fichier} — " + " ; ".join(soucis) if soucis else None)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


@dataclass
class Config:
    """Toute la configuration de l'application en un seul objet immuable."""

    base_url: str = DEFAULT_BASE_URL
    secours: list[str] = field(default_factory=list)  # autres serveurs, dans l'ordre
    model: str = ""
    api_key: str = ""
    system: str = DEFAULT_SYSTEM

    voice: str = DEFAULT_VOICE
    lang: str = DEFAULT_LANG
    speed: float = DEFAULT_SPEED
    device: str = "auto"  # auto | cuda | cpu
    output_device: int | None = None
    tts: bool = True
    cache: bool = True  # réserve disque des phrases déjà synthétisées
    tts_url: str = ""  # serveur TTS distant (v1.0) ; vide = synthèse locale

    temperature: float = 0.7
    max_tokens: int = 0  # 0 = laisser le serveur décider
    timeout: float = 300.0  # un modèle local peut être lent au 1er token
    profil: str = ""  # profil de prompt système à charger au démarrage

    # entrée vocale
    micro: bool = False  # --micro : mains libres (écoute après chaque réponse)
    stt_modele: str = "small"
    stt_device: str = "auto"  # auto | cuda | cpu
    stt_langue: str = "fr"  # "" = détection automatique
    micro_device: int | str | None = None

    show_stats: bool = False
    history_limit: int = 24  # nb de messages (hors système) gardés en contexte

    @classmethod
    def from_env(cls, base: dict | None = None) -> "Config":
        """Configuration depuis le fichier TOML, puis les variables d'environnement.

        Le TOML sert de **valeur par défaut** : chaque variable d'environnement le
        surclasse. L'ordre final est donc arguments de ligne de commande > variables
        d'environnement > fichier TOML > valeurs codées en dur.

        ``base`` permet de passer les réglages déjà lus (et de n'avertir qu'une fois
        sur un fichier abîmé) ; sinon le fichier est lu ici.
        """
        if base is None:
            base = lire_config()[0]

        # Défauts issus de la définition de Config : évite de recopier les mêmes
        # littéraux (0.7, 300.0…) ici et dans la dataclass, où ils finiraient par diverger.
        defauts = cls()

        def env(nom: str, cle: str, defaut):
            """Variable d'environnement, sinon le TOML, sinon le défaut."""
            return _env(nom, str(base.get(cle, defaut)))

        return cls(
            base_url=env("VOICECHAT_BASE_URL", "base_url", DEFAULT_BASE_URL).rstrip("/"),
            secours=_liste_urls(
                os.environ.get("VOICECHAT_SECOURS") or base.get("secours") or ""
            ),
            model=env("VOICECHAT_MODEL", "model", ""),
            api_key=env("VOICECHAT_API_KEY", "api_key", ""),
            system=env("VOICECHAT_SYSTEM", "system", DEFAULT_SYSTEM),
            voice=env("VOICECHAT_VOICE", "voice", DEFAULT_VOICE),
            lang=env("VOICECHAT_LANG", "lang", DEFAULT_LANG),
            speed=_env_float("VOICECHAT_SPEED", float(base.get("speed", 1.0))),
            device=env("VOICECHAT_DEVICE", "device", "auto").lower(),
            tts=_env_bool("VOICECHAT_TTS", bool(base.get("tts", True))),
            cache=_env_bool("VOICECHAT_CACHE", bool(base.get("cache", True))),
            tts_url=env("VOICECHAT_TTS_URL", "tts_url", "").rstrip("/"),
            profil=env("VOICECHAT_PROFIL", "profil", ""),
            micro=_env_bool("VOICECHAT_MICRO", bool(base.get("micro", False))),
            stt_modele=env("VOICECHAT_STT_MODELE", "stt_modele", "small"),
            stt_device=env("VOICECHAT_STT_DEVICE", "stt_device", "auto").lower(),
            stt_langue=env("VOICECHAT_STT_LANGUE", "stt_langue", "fr"),
            micro_device=env("VOICECHAT_MICRO_DEVICE", "micro_device", "") or None,
            # Ces trois-là n'étaient réglables ni par l'environnement ni par le fichier :
            # seuls les arguments les touchaient. `[modeles.x] temperature = 0.2`
            # fonctionnait, `[voicechat] temperature = 0.9` non — incohérent.
            temperature=_env_float(
                "VOICECHAT_TEMPERATURE", float(base.get("temperature", defauts.temperature))
            ),
            max_tokens=int(
                _env_float("VOICECHAT_MAX_TOKENS", float(base.get("max_tokens", defauts.max_tokens)))
            ),
            timeout=_env_float(
                "VOICECHAT_TIMEOUT", float(base.get("timeout", defauts.timeout))
            ),
        )

    def pour_modele(self, modele: str) -> tuple["Config", dict]:
        """Applique une éventuelle section ``[modeles."<nom>"]`` du fichier TOML.

        Renvoie la config ajustée et les réglages appliqués (pour pouvoir le dire).
        Deux noms sont essayés : le nom complet renvoyé par le serveur (llama.cpp
        renvoie un chemin) et sa forme courte.
        """
        if not modele:
            return self, {}
        base = lire_config()[0]
        table = base.get("modeles") or {}
        if not isinstance(table, dict):
            return self, {}
        for nom in (modele, nom_court(modele)):
            section = table.get(nom)
            if isinstance(section, dict):
                connues = {champ.name for champ in fields(Config)}
                propre = {c: v for c, v in section.items() if c in connues}
                return (self.with_overrides(**propre) if propre else self), propre
        return self, {}

    def with_overrides(self, **kwargs) -> "Config":
        """Retourne une copie en ignorant les valeurs None (args non fournis)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    @property
    def cibles(self) -> list[str]:
        """Serveurs à essayer, dans l'ordre : la cible principale, puis les secours.

        Dédoublonné, parce qu'un secours identique à la principale ferait perdre
        deux fois le même délai d'attente.
        """
        vues: list[str] = []
        for url in [self.base_url, *self.secours]:
            propre = (url or "").strip().rstrip("/")
            if propre and propre not in vues:
                vues.append(propre)
        return vues

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"
