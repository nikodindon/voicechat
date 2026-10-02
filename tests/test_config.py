"""Tests de la configuration (env + surcharges)."""

import os

from voicechat.config import DEFAULT_BASE_URL, Config, nom_court


def test_defauts():
    cfg = Config()
    assert cfg.base_url == DEFAULT_BASE_URL
    assert cfg.chat_url == f"{DEFAULT_BASE_URL}/chat/completions"
    assert cfg.models_url == f"{DEFAULT_BASE_URL}/models"
    assert cfg.voice == "ff_siwis"


def test_env_prioritaire(monkeypatch):
    monkeypatch.setenv("VOICECHAT_BASE_URL", "http://127.0.0.1:8099/v1")
    monkeypatch.setenv("VOICECHAT_VOICE", "af_heart")
    monkeypatch.setenv("VOICECHAT_SPEED", "1.3")
    monkeypatch.setenv("VOICECHAT_TTS", "0")
    cfg = Config.from_env()
    assert cfg.base_url == "http://127.0.0.1:8099/v1"
    assert cfg.voice == "af_heart"
    assert cfg.speed == 1.3
    assert cfg.tts is False


def test_surcharge_ignore_none():
    cfg = Config().with_overrides(voice="af_heart", speed=None)
    assert cfg.voice == "af_heart"
    assert cfg.speed == 1.0  # None n'écrase pas la valeur par défaut


def test_env_speed_invalide_ne_plante_pas(monkeypatch):
    monkeypatch.setenv("VOICECHAT_SPEED", "vite")
    assert Config.from_env().speed == 1.0


def test_url_sans_slash_final(monkeypatch):
    monkeypatch.setenv("VOICECHAT_BASE_URL", "http://hote:8080/v1/")
    cfg = Config.from_env()
    assert cfg.chat_url == "http://hote:8080/v1/chat/completions"


# ------------------------------------------------------- nom d'affichage
def test_nom_court_retire_le_chemin_et_l_extension():
    """llama.cpp renvoie un chemin complet : on n'affiche que le nom du fichier."""
    assert (
        nom_court("/mnt/data/sdc2/models/Ornith-1.5-35B-A3B-APEX-i-mini.gguf")
        == "Ornith-1.5-35B-A3B-APEX-i-mini"
    )


def test_nom_court_laisse_un_nom_simple_intact():
    assert nom_court("Ornith-1.5-35B-A3B-APEX-i-mini.gguf") == "Ornith-1.5-35B-A3B-APEX-i-mini"
    assert nom_court("mistral") == "mistral"


def test_nom_court_sans_modele():
    assert nom_court("") == "?"


def test_le_modele_envoye_reste_le_chemin_complet():
    """Le raccourci est purement cosmétique : l'identifiant transmis ne change pas."""
    cfg = Config(model="/mnt/data/sdc2/models/machin.gguf")
    assert cfg.model == "/mnt/data/sdc2/models/machin.gguf"
    assert nom_court(cfg.model) == "machin"
    assert cfg.chat_url.endswith("/chat/completions")
