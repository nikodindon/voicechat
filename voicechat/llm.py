"""Client minimal pour une API OpenAI-compatible, en streaming SSE.

Volontairement sans dépendance (urllib) : un seul fichier à comprendre, et
aucun paquet à mettre à jour côté réseau.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator


class LLMError(RuntimeError):
    """Erreur réseau ou protocole, avec un message lisible par un humain."""


@dataclass
class Usage:
    """Petites mesures pour /stats."""

    first_token_s: float = 0.0
    total_s: float = 0.0
    chars: int = 0

    @property
    def chars_per_s(self) -> float:
        return self.chars / self.total_s if self.total_s > 0 else 0.0


def _headers(api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def list_models(base_url: str, api_key: str = "", timeout: float = 10.0) -> list[str]:
    """Interroge /v1/models. Lève LLMError avec un message clair en cas d'échec."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers=_headers(api_key), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Certains serveurs n'exposent pas /v1/models : ce n'est pas fatal.
        raise LLMError(f"/v1/models a répondu HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(_net_msg(url, exc)) from exc
    except (TimeoutError, OSError) as exc:
        raise LLMError(_net_msg(url, exc)) from exc

    items = data.get("data") or []
    return [m.get("id", "") for m in items if isinstance(m, dict)]


def _net_msg(url: str, exc: Exception) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return (
        f"Connexion impossible à {host} ({exc}).\n"
        "  → la machine distante est probablement éteinte, ou le serveur LLM "
        "n'écoute pas sur 0.0.0.0.\n"
        "  → vérifier : tailscale status ; ou lancer le serveur avec --host 0.0.0.0"
    )


def stream_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 0,
    timeout: float = 300.0,
) -> Iterator[str]:
    """Génère les morceaux de texte de la réponse, au fil de l'eau.

    Utilisation :
        for morceau in stream_chat(...):
            ...
    Lève LLMError si le transport échoue avant ou pendant le flux.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens

    url = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(api_key),
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        corps = ""
        try:
            corps = exc.read().decode("utf-8", "replace")[:400]
        except Exception:  # pragma: no cover - lecture best effort
            pass
        if exc.code == 404:
            raise LLMError(
                f"HTTP 404 sur {url} — l'URL ne ressemble pas à une API "
                f"OpenAI-compatible (attendu : .../v1/chat/completions).\n{corps}"
            ) from exc
        raise LLMError(f"HTTP {exc.code} de {url}\n{corps}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(_net_msg(url, exc)) from exc
    except (TimeoutError, OSError) as exc:
        raise LLMError(_net_msg(url, exc)) from exc

    with resp:
        try:
            for raw in resp:
                ligne = raw.decode("utf-8", "replace").strip()
                if not ligne or ligne.startswith(":"):
                    continue  # keep-alive
                if not ligne.startswith("data:"):
                    continue
                charge = ligne[5:].strip()
                if charge == "[DONE]":
                    return
                try:
                    obj = json.loads(charge)
                except json.JSONDecodeError:
                    continue
                for choix in obj.get("choices") or []:
                    delta = choix.get("delta") or {}
                    # Les modèles « raisonneurs » séparent le raisonnement du texte.
                    if delta.get("reasoning_content"):
                        continue
                    morceau = delta.get("content")
                    if morceau:
                        yield morceau
        except (TimeoutError, OSError) as exc:
            raise LLMError(f"Flux interrompu : {exc}") from exc


def stream_with_usage(**kwargs) -> tuple[Iterator[str], Usage]:
    """Comme stream_chat mais renvoie aussi un objet Usage rempli au fil de l'eau.

    Le générateur met à jour ``usage`` ; l'appelant peut le lire après épuisement.
    """
    usage = Usage()
    debut = time.monotonic()

    def gen() -> Iterator[str]:
        for morceau in stream_chat(**kwargs):
            if usage.first_token_s == 0.0:
                usage.first_token_s = time.monotonic() - debut
            usage.chars += len(morceau)
            yield morceau
        usage.total_s = time.monotonic() - debut

    return gen(), usage
