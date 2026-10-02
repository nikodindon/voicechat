"""Client minimal pour une API OpenAI-compatible, en streaming SSE.

Volontairement sans dépendance (urllib) : un seul fichier à comprendre, et
aucun paquet à mettre à jour côté réseau.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Iterator


class LLMError(RuntimeError):
    """Erreur réseau ou protocole, avec un message lisible par un humain."""


# Politique de reprise. Le délai double à chaque essai : 0,5 s puis 1 s puis 2 s.
# Trois essais couvrent le cas courant (serveur en cours de redémarrage : llama.cpp
# met quelques secondes à réécouter) sans transformer une panne franche en attente
# interminable. Le total reste sous 4 s.
ESSAIS_DEFAUT = 3
DELAI_DEFAUT = 0.5
DELAI_MAX = 8.0


@dataclass
class Connexion:
    """Quel serveur a réellement répondu, et ce qui a échoué avant.

    Rempli par `stream_chat`. Sans cette trace, une bascule vers un serveur de
    secours serait invisible : l'utilisateur verrait une réponse arriver sans
    savoir qu'elle ne vient pas de la machine attendue.
    """

    url: str = ""
    tentatives: int = 0
    echecs: list[str] = field(default_factory=list)

    @property
    def bascule(self) -> bool:
        """Vrai si la cible principale n'a pas répondu et qu'on a changé."""
        return bool(self.echecs)

    def resume(self) -> str:
        if not self.bascule:
            return self.url
        return f"{self.url} (après échec de {len(self.echecs)} cible(s))"


def _cibles(base_url: str | Sequence[str]) -> list[str]:
    """Normalise en liste d'URL sans doublon, dans l'ordre donné."""
    brut = [base_url] if isinstance(base_url, str) else list(base_url)
    vues: list[str] = []
    for cible in brut:
        propre = (cible or "").strip().rstrip("/")
        if propre and propre not in vues:
            vues.append(propre)
    return vues


def _temporaire(exc: BaseException) -> bool:
    """Vrai si l'erreur peut disparaître en réessayant.

    Une connexion refusée ou un timeout sont transitoires (le serveur redémarre).
    Un 404 ou un 400 sont définitifs : insister ne changerait rien et ferait
    perdre du temps à l'utilisateur.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


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


def contexte_du_serveur(base_url: str, api_key: str = "", timeout: float = 10.0) -> int:
    """Taille du contexte annoncée par le serveur, en tokens. 0 si inconnue.

    llama.cpp expose ``/props``, qui donne ``n_ctx``. Vérifié sur le serveur du projet
    (4 slots parallèles) : chaque slot annonce bien 32768, donc une conversation
    dispose de tout le contexte — ce n'est pas divisé par le nombre de slots.

    Cette information manquait au client, qui comptait les messages **à l'aveugle** :
    mesuré, 25 messages (l'ancienne limite) ne pesaient que 1 646 tokens, soit 5 % d'un
    contexte de 32768. Autrement dit, on jetait des messages vingt fois trop tôt.

    Aucune exception ne sort d'ici : un serveur qui ne connaît pas ``/props`` (un autre
    moteur OpenAI-compatible, par exemple) laisse simplement la valeur à 0, et le client
    retombe sur son ancien comportement.
    """
    # /props vit à la racine du serveur, pas sous /v1.
    racine = base_url.rstrip("/")
    if racine.endswith("/v1"):
        racine = racine[: -len("/v1")]
    try:
        req = urllib.request.Request(
            f"{racine}/props", headers=_headers(api_key), method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            donnees = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        return 0  # information facultative : jamais une raison d'échouer

    if not isinstance(donnees, dict):
        return 0
    defauts = donnees.get("default_generation_settings")
    if isinstance(defauts, dict):
        n_ctx = defauts.get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            return n_ctx
    n_ctx = donnees.get("n_ctx")
    return n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else 0


def premier_serveur(
    base_url: str | Sequence[str],
    api_key: str = "",
    timeout: float = 10.0,
) -> tuple[list[str], str]:
    """Modèles du premier serveur qui répond, et son URL.

    Sert au démarrage : on veut la liste des modèles *et* savoir quelle machine a
    répondu, pour pouvoir l'afficher. Lève LLMError si aucune cible ne répond.
    """
    derniere: LLMError | None = None
    for cible in _cibles(base_url):
        try:
            return list_models(cible, api_key, timeout), cible
        except LLMError as exc:
            derniere = exc
    raise derniere or LLMError("aucun serveur n'a répondu")


def _ouvrir_flux(
    url: str,
    payload: dict,
    api_key: str,
    timeout: float,
    *,
    essais: int = ESSAIS_DEFAUT,
    delai: float = DELAI_DEFAUT,
    on_essai: Callable[[int, float, str], None] | None = None,
):
    """Ouvre le flux SSE, en réessayant les pannes temporaires.

    Deux raisons de réessayer :

    * **le serveur redémarre** — une connexion refusée pendant deux secondes est le
      cas le plus courant, et il n'y a rien d'autre à faire qu'attendre ;
    * certains serveurs OpenAI-compatibles rejettent ``stream_options`` (HTTP 400) :
      on réessaie alors une fois sans ce champ — on perd les compteurs, pas la réponse.

    Un 404 ou un 400 hors ``stream_options`` n'est **pas** réessayé : insister ne
    changerait rien. ``on_essai(numero, attente, raison)`` est appelé avant chaque
    nouvelle attente : un retry silencieux ressemble sinon à un blocage.
    """
    derniere: LLMError | None = None
    attente = delai

    for numero in range(1, max(1, essais) + 1):
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
                if not _temporaire(exc):
                    raise LLMError(f"HTTP {exc.code} de {url}\n{detail}") from exc
                derniere = LLMError(f"HTTP {exc.code} de {url}\n{detail}")
                break  # temporaire : on retente après une attente
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if not _temporaire(exc):
                    raise LLMError(_net_msg(url, exc)) from exc
                derniere = LLMError(_net_msg(url, exc))
                break
        else:
            # Les deux formes ont été refusées : insister n'a pas de sens.
            raise derniere or LLMError(f"Aucune requête n'a abouti vers {url}")

        if numero < essais:
            if on_essai:
                on_essai(numero, attente, str(derniere))
            time.sleep(attente)
            attente = min(attente * 2, DELAI_MAX)

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
    base_url: str | Sequence[str],
    model: str,
    messages: list[dict],
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 0,
    timeout: float = 300.0,
    usage: Usage | None = None,
    connexion: Connexion | None = None,
    essais: int = ESSAIS_DEFAUT,
    delai: float = DELAI_DEFAUT,
    on_essai: Callable[[int, float, str], None] | None = None,
) -> Iterator[str]:
    """Génère les morceaux de texte de la réponse, au fil de l'eau.

    Utilisation :
        for morceau in stream_chat(...):
            ...
    Si ``usage`` est fourni, il est rempli avec les compteurs renvoyés par le serveur.
    ``base_url`` peut être une liste : les serveurs sont essayés dans l'ordre, et
    ``connexion`` garde la trace de celui qui a répondu (et des échecs précédents).

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

    # On essaie les serveurs dans l'ordre. La bascule n'a lieu que si la connexion
    # n'a pas pu être ouverte : une fois le premier morceau reçu, changer de serveur
    # dupliquerait du texte déjà affiché. Un flux qui casse en cours de route est
    # donc signalé, jamais rejoué en silence.
    reponse = None
    derniere: LLMError | None = None
    for cible in _cibles(base_url):
        url = f"{cible}/chat/completions"
        try:
            reponse = _ouvrir_flux(
                url, payload, api_key, timeout,
                essais=essais, delai=delai, on_essai=on_essai,
            )
        except LLMError as exc:
            derniere = exc
            if connexion is not None:
                connexion.echecs.append(f"{cible} : {exc}")
            continue
        if connexion is not None:
            connexion.url = cible
        break

    if reponse is None:
        raise derniere or LLMError("aucun serveur n'a répondu")

    with reponse:
        try:
            for raw in reponse:
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
            # Une coupure en cours de flux n'est PAS rejouée : le texte déjà affiché
            # le serait aussi. On le dit, et l'appelant garde la réponse partielle.
            raise LLMError(f"Flux interrompu (serveur {cible}) : {exc}") from exc


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
