"""Configuration : variables d'environnement + ligne de commande fusionnées."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

DEFAULT_BASE_URL = "http://100.108.224.60:8080/v1"
DEFAULT_VOICE = "ff_siwis"  # français féminin
DEFAULT_LANG = "f"
DEFAULT_SYSTEM = (
    "Tu es un assistant francophone concis et précis. "
    "Réponds en phrases courtes et parlées, sans listes à puces ni markdown."
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = True) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


@dataclass
class Config:
    """Toute la configuration de l'application en un seul objet immuable."""

    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_key: str = ""
    system: str = DEFAULT_SYSTEM

    voice: str = DEFAULT_VOICE
    lang: str = DEFAULT_LANG
    speed: float = 1.0
    device: str = "auto"  # auto | cuda | cpu
    output_device: int | None = None
    tts: bool = True

    temperature: float = 0.7
    max_tokens: int = 0  # 0 = laisser le serveur décider
    timeout: float = 300.0  # un modèle local peut être lent au 1er token

    show_stats: bool = False
    history_limit: int = 24  # nb de messages (hors système) gardés en contexte

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            base_url=_env("VOICECHAT_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=_env("VOICECHAT_MODEL"),
            api_key=_env("VOICECHAT_API_KEY"),
            system=_env("VOICECHAT_SYSTEM", DEFAULT_SYSTEM),
            voice=_env("VOICECHAT_VOICE", DEFAULT_VOICE),
            lang=_env("VOICECHAT_LANG", DEFAULT_LANG),
            speed=_env_float("VOICECHAT_SPEED", 1.0),
            device=_env("VOICECHAT_DEVICE", "auto").lower(),
            tts=_env_bool("VOICECHAT_TTS", True),
        )

    def with_overrides(self, **kwargs) -> "Config":
        """Retourne une copie en ignorant les valeurs None (args non fournis)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"
