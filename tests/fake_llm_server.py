"""Faux serveur OpenAI-compatible, pour valider le client sans le serveur réel.

Usage :
    python3 tests/fake_llm_server.py 8099

Expose /v1/models et /v1/chat/completions (streaming SSE), avec un petit délai
entre les tokens pour reproduire un modèle local réel.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELES = ["fake-local-model"]
REPONSE = (
    "Bonjour ! Voici une réponse de test, découpée en plusieurs phrases. "
    "Elle sert à vérifier que le découpage en phrases fonctionne. "
    "Et que chaque phrase part bien vers la synthèse vocale au fur et à mesure. "
    "Fin du message."
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence par défaut
        if "-v" in sys.argv:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            corps = json.dumps(
                {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELES]}
            ).encode()
            self._send(200, corps, "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, b'{"error":"not found"}', "application/json")
            return

        taille = int(self.headers.get("Content-Length", 0))
        demande = json.loads(self.rfile.read(taille) or b"{}")
        modele = demande.get("model", MODELES[0])
        stream = bool(demande.get("stream"))

        if not stream:
            corps = json.dumps(
                {
                    "choices": [
                        {"message": {"role": "assistant", "content": REPONSE}, "index": 0}
                    ]
                }
            ).encode()
            self._send(200, corps, "application/json")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            for mot in REPONSE.split(" "):
                charge = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": modele,
                    "choices": [{"index": 0, "delta": {"content": mot + " "}}],
                }
                self.wfile.write(f"data: {json.dumps(charge)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.03)  # simule un modèle lent
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # le client a coupé (Ctrl+C) : normal

    # ---------------------------------------------------------------- utilitaire
    def _send(self, code: int, corps: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"faux serveur LLM sur http://127.0.0.1:{port}/v1  (Ctrl+C pour arrêter)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
