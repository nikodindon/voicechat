"""Test d'intégration du client LLM : faux serveur OpenAI dans un thread.

Ne touche ni au réseau externe ni à la carte son.
"""

from __future__ import annotations

import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_llm_server import Handler  # noqa: E402

from voicechat.llm import LLMError, list_models, stream_with_usage  # noqa: E402
from voicechat.text import split_sentences  # noqa: E402


@pytest.fixture()
def serveur():
    """Lance le faux serveur sur un port libre et le coupe après le test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def test_liste_les_modeles(serveur):
    assert list_models(serveur) == ["fake-local-model"]


def test_streaming_et_decoupage_en_phrases(serveur):
    flux, usage = stream_with_usage(
        base_url=serveur,
        model="fake-local-model",
        messages=[{"role": "user", "content": "bonjour"}],
    )

    tampon = ""
    phrases: list[str] = []
    for morceau in flux:
        tampon += morceau
        nouvelles, tampon = split_sentences(tampon)
        phrases.extend(nouvelles)
    if tampon.strip():
        phrases.append(tampon.strip())

    assert len(phrases) >= 3, f"attendu plusieurs phrases, obtenu {phrases}"
    assert phrases[0] == "Bonjour !"
    assert phrases[-1].endswith("message.")
    assert usage.chars > 0
    assert usage.total_s > 0
    assert usage.first_token_s > 0


def test_erreur_reseau_message_lisible():
    # Port fermé : doit lever LLMError avec un message compréhensible, pas une trace brute.
    with pytest.raises(LLMError) as info:
        list_models("http://127.0.0.1:1/v1", timeout=2.0)
    message = str(info.value)
    assert "Connexion impossible" in message
    assert "tailscale" in message
