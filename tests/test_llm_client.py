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

from voicechat.llm import LLMError, list_models, stream_with_usage, Usage  # noqa: E402
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


# ------------------------------------------------------- compteurs de tokens
def test_streaming_remonte_les_tokens(serveur):
    """Le faux serveur renvoie usage + timings comme llama.cpp : on doit les capter."""
    flux, usage = stream_with_usage(
        base_url=serveur,
        model="fake-local-model",
        messages=[{"role": "user", "content": "bonjour"}],
    )
    for _ in flux:
        pass

    assert usage.completion_tokens > 0
    assert usage.prompt_tokens == 42
    assert usage.server_tok_s == 22.5
    assert usage.tok_s > 0
    assert usage.first_token_s > 0
    assert usage.total_s > 0


def test_stats_avec_tokens():
    usage = Usage(first_token_s=1.0, total_s=3.0, chars=100, prompt_tokens=42, completion_tokens=20)
    # 20 tokens après le 1er, sur 2 s de génération effective
    assert usage.tok_s == pytest.approx(10.0)
    resume = usage.resume()
    assert "20 tok" in resume
    assert "+42 prompt" in resume
    assert "10.0 tok/s" in resume


def test_stats_sans_tokens_replient_sur_les_caracteres():
    """Un serveur sans stream_options doit rester lisible, pas afficher « 0 tok »."""
    usage = Usage(first_token_s=0.5, total_s=2.0, chars=200)
    resume = usage.resume()
    assert "200 car." in resume
    assert "tok/s" not in resume, "pas de débit en tokens quand le serveur n'en fournit pas"


def test_stats_sans_premier_token_ne_divisent_pas_par_zero():
    usage = Usage(total_s=0.0, chars=0, completion_tokens=5, first_token_s=0.0)
    assert usage.tok_s == 0.0
    assert usage.chars_per_s == 0.0
    assert "5 tok" in usage.resume()


def test_debit_utilise_les_timings_serveur_si_pas_de_mesure_locale():
    usage = Usage(first_token_s=1.0, total_s=1.0, completion_tokens=8, server_tok_s=22.5)
    assert usage.tok_s == 0.0  # pas de temps après le 1er token
    assert "22.5 tok/s" in usage.resume()


def test_interruption_renseigne_quand_meme_la_duree(serveur):
    """Après un Ctrl+C, usage.total_s doit être renseigné (finally du générateur)."""
    flux, usage = stream_with_usage(
        base_url=serveur,
        model="fake-local-model",
        messages=[{"role": "user", "content": "bonjour"}],
    )
    next(flux)      # premier morceau
    flux.close()    # simule une interruption
    assert usage.total_s > 0
    assert usage.chars > 0
