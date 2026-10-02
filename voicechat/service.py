"""Plomberie HTTP commune aux services (TTS, STT, web).

Ces quelques méthodes étaient dans le serveur TTS (`distant.py`). Le service STT allait
les recopier — et une copie de ce genre finit toujours par diverger : le jour où l'on
corrige l'encodage des accents d'un côté, l'autre garde le défaut.

Ce qui vit ici :

* `log_message` **silencieux** : un service ne doit pas polluer le terminal de celui qui
  l'utilise, ni remplir les journaux de lignes d'accès ;
* `_envoyer` / `_json` / `_erreur` : les réponses, avec le charset déclaré dès qu'il
  s'agit de texte — sans quoi « synthétiseur non chargé » s'affiche en `\\u00e9` au curl ;
* `_lire_json` : la lecture d'un corps JSON, avec un message d'erreur exploitable.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

TITRE_SERVICE = "voicechat"


class HandlerService(BaseHTTPRequestHandler):
    """Socle des gestionnaires de service : réponses, erreurs, silence."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002 (nom imposé)
        """Silence : le service ne pollue pas le terminal de ses clients."""
        return

    # ------------------------------------------------------------------ réponses
    def _envoyer(self, code: int, corps: bytes, ctype: str) -> None:
        self.send_response(code)
        # Le charset vaut pour le texte (JSON, HTML, flux d'évènements). L'annoncer sur
        # audio/wav serait faux : d'où le test sur le type.
        if ctype.startswith(("application/json", "text/")):
            ctype = f"{ctype}; charset=utf-8"
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def _json(self, code: int, donnees: dict) -> None:
        # ensure_ascii=False : sinon les accents partent en \u00e9, illisibles au curl.
        corps = json.dumps(donnees, ensure_ascii=False).encode("utf-8")
        self._envoyer(code, corps, "application/json")

    def _erreur(self, code: int, message: str) -> None:
        self._json(code, {"erreur": message})

    # -------------------------------------------------------------------- lecture
    def _lire_corps(self) -> bytes:
        taille = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(taille) if taille else b""

    def _lire_json(self) -> dict | None:
        """Corps JSON, ou ``None`` si illisible (l'erreur est déjà envoyée au client)."""
        try:
            donnees = json.loads(self._lire_corps() or b"{}")
        except json.JSONDecodeError:
            self._erreur(400, "corps JSON illisible")
            return None
        if not isinstance(donnees, dict):
            self._erreur(400, "le corps doit être un objet JSON")
            return None
        return donnees
