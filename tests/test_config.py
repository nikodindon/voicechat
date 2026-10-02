"""Tests de la configuration (env + surcharges)."""

import os

from voicechat.config import DEFAULT_BASE_URL, Config


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
