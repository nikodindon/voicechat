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
    """Mesures d'un échange, remplies au fil de l'eau.

    Les compteurs de tokens viennent du serveur (``usage``) ; s'il ne les fournit pas,
    on retombe sur un comptage de caractères. ``server_tok_s`` est le débit annoncé
    par llama.cpp (``timings.predicted_per_second``), plus juste que l'horloge locale.
    """

    first_token_s: float = 0.0
    total_s: float = 0.0
    chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    server_tok_s: float = 0.0

    @property
    def chars_per_s(self) -> float:
        return self.chars / self.total_s if self.total_s > 0 else 0.0

    @property
    def tok_s(self) -> float:
        """Débit de génération : tokens produits / temps écoulé après le 1er token.

        On exclut le délai du premier token, sans quoi un long prompt ferait
        paraître le modèle plus lent qu'il ne l'est.
        """
        duree = self.total_s - self.first_token_s
        if self.completion_tokens and duree > 0:
            return self.completion_tokens / duree
        return 0.0

    def resume(self) -> str:
        """Ligne de statistiques lisible (utilisée par /stats et --debug)."""
        morceaux = [f"1er token {self.first_token_s:.2f} s"]
        if self.completion_tokens:
            detail = f"{self.completion_tokens} tok"
            if self.prompt_tokens:
                detail += f" (+{self.prompt_tokens} prompt)"
            morceaux.append(detail)
            vitesse = self.tok_s or self.server_tok_s
            if vitesse:
                morceaux.append(f"{vitesse:.1f} tok/s")
        else:
            duree = self.total_s or 0.0
            morceaux.append(f"{self.chars} car. en {duree:.2f} s ({self.chars_per_s:.1f} car/s)")
        return " | ".join(morceaux)


def _headers(api_key: str) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _net_msg(url: str, exc: Exception) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return (
        f"Connexion impossible à {host} ({exc}).\n"
        "  → la machine distante est probablement éteinte, ou le serveur LLM "
        "n'écoute pas sur 0.0.0.0.\n"
        "  → vérifier : tailscale status ; ou lancer le serveur avec --host 0.0.0.0"
    )


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


def _ouvrir_flux(url: str, payload: dict, api_key: str, timeout: float):
    """Ouvre le flux SSE, en se rabattant si le serveur refuse ``stream_options``.

    Beaucoup de serveurs OpenAI-compatibles (vLLM anciens, Ollama…) rejettent ce champ
    avec un 400. On réessaie alors une fois sans lui : on perd les tokens, pas la réponse.
    """
    derniere: LLMError | None = None

    for avec_usage in (True, False):
        corps = dict(payload)
        if avec_usage:
            corps["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(
            url,
            data=json.dumps(corps).encode("utf-8"),
            headers=_headers(api_key),
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:  # pragma: no cover - lecture best effort
                pass
            if avec_usage and exc.code == 400:
                derniere = LLMError(f"HTTP 400 de {url}\n{detail}")
                continue  # serveur sans stream_options : on retente sans
            if exc.code == 404:
                raise LLMError(
                    f"HTTP 404 sur {url} — l'URL ne ressemble pas à une API "
                    f"OpenAI-compatible (attendu : .../v1/chat/completions).\n{detail}"
                ) from exc
            raise LLMError(f"HTTP {exc.code} de {url}\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(_net_msg(url, exc)) from exc
        except (TimeoutError, OSError) as exc:
            raise LLMError(_net_msg(url, exc)) from exc

    raise derniere or LLMError(f"Aucune requête n'a abouti vers {url}")


def _absorber_usage(usage: Usage, obj: dict) -> None:
    """Récupère les compteurs de tokens et les timings du serveur, s'ils sont là."""
    bloc = obj.get("usage")
    if isinstance(bloc, dict):
        try:
            usage.prompt_tokens = int(bloc.get("prompt_tokens") or usage.prompt_tokens)
            usage.completion_tokens = int(bloc.get("completion_tokens") or usage.completion_tokens)
        except (TypeError, ValueError):
            pass

    timings = obj.get("timings")  # spécifique llama.cpp
    if isinstance(timings, dict):
        try:
            vitesse = float(timings.get("predicted_per_second") or 0.0)
            if vitesse > 0:
                usage.server_tok_s = vitesse
        except (TypeError, ValueError):
            pass


def stream_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 0,
    timeout: float = 300.0,
    usage: Usage | None = None,
) -> Iterator[str]:
    """Génère les morceaux de texte de la réponse, au fil de l'eau.

    Utilisation :
        for morceau in stream_chat(...):
            ...
    Si ``usage`` est fourni, il est rempli avec les compteurs renvoyés par le serveur.
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
    resp = _ouvrir_flux(url, payload, api_key, timeout)

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

                if usage is not None:
                    _absorber_usage(usage, obj)

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

    Le générateur met à jour ``usage`` ; l'appelant peut le lire après épuisement —
    y compris s'il a interrompu la lecture en cours de route.
    """
    usage = Usage()
    debut = time.monotonic()

    def gen() -> Iterator[str]:
        try:
            for morceau in stream_chat(usage=usage, **kwargs):
                if usage.first_token_s == 0.0:
                    usage.first_token_s = time.monotonic() - debut
                usage.chars += len(morceau)
                yield morceau
        finally:
            # Aussi en cas d'interruption : les stats doivent rester lisibles.
            usage.total_s = time.monotonic() - debut

    return gen(), usage
