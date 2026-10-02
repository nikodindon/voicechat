"""Tests de la configuration (env + surcharges + fichier TOML)."""

import os

import pytest

from voicechat.config import DEFAULT_BASE_URL, Config, lire_config, nom_court


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


# --------------------------------------------------------- fichier TOML (v1.0)
def ecrire_toml(tmp_path, contenu: str, monkeypatch):
    """Pose un config.toml isolé et le désigne par VOICECHAT_CONFIG."""
    chemin = tmp_path / "config.toml"
    chemin.write_text(contenu, encoding="utf-8")
    monkeypatch.setenv("VOICECHAT_CONFIG", str(chemin))
    # On neutralise les variables d'environnement qui viendraient surclasser le fichier.
    for nom in (
        "VOICECHAT_BASE_URL",
        "VOICECHAT_VOICE",
        "VOICECHAT_SPEED",
        "VOICECHAT_TTS",
        "VOICECHAT_SECOURS",
    ):
        monkeypatch.delenv(nom, raising=False)
    return chemin


def test_toml_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICECHAT_CONFIG", str(tmp_path / "inexistant.toml"))
    reglages, avertissement = lire_config()
    assert reglages == {} and avertissement is None


def test_toml_fournit_les_defauts(tmp_path, monkeypatch):
    ecrire_toml(
        tmp_path,
        '[voicechat]\nvoice = "af_heart"\nspeed = 1.25\ntts = false\n',
        monkeypatch,
    )
    cfg = Config.from_env()
    assert cfg.voice == "af_heart"
    assert cfg.speed == pytest.approx(1.25)
    assert cfg.tts is False


def test_env_surclasse_le_toml(tmp_path, monkeypatch):
    """L'ordre promis : arguments > environnement > TOML > défauts."""
    ecrire_toml(tmp_path, '[voicechat]\nvoice = "af_heart"\n', monkeypatch)
    monkeypatch.setenv("VOICECHAT_VOICE", "if_sara")
    assert Config.from_env().voice == "if_sara"


def test_toml_liste_de_secours(tmp_path, monkeypatch):
    ecrire_toml(
        tmp_path,
        '[voicechat]\nsecours = ["http://a:8080/v1", "http://b:8080/v1"]\n',
        monkeypatch,
    )
    cfg = Config.from_env()
    assert cfg.cibles == [
        DEFAULT_BASE_URL,
        "http://a:8080/v1",
        "http://b:8080/v1",
    ]


def test_toml_cle_inconnue_est_signalee(tmp_path, monkeypatch):
    """Une faute de frappe doit être dite : sinon le réglage ne s'applique jamais."""
    ecrire_toml(tmp_path, '[voicechat]\nvoix = "af_heart"\n', monkeypatch)
    reglages, avertissement = lire_config()
    assert avertissement is not None and "voix" in avertissement
    assert "voix" not in reglages, "une clé inconnue ne doit pas devenir un réglage"


def test_toml_sans_section_voicechat(tmp_path, monkeypatch):
    ecrire_toml(tmp_path, '[autre]\nx = 1\n', monkeypatch)
    _, avertissement = lire_config()
    assert avertissement is not None and "[voicechat]" in avertissement


def test_toml_abime_ne_plante_pas(tmp_path, monkeypatch):
    """Un fichier de configuration cassé ne doit pas empêcher de démarrer."""
    ecrire_toml(tmp_path, "[voicechat\nceci n'est pas du toml", monkeypatch)
    reglages, avertissement = lire_config()
    assert reglages == {}
    assert avertissement is not None and "illisible" in avertissement


def test_toml_parse_quand_tomllib_absent(tmp_path, monkeypatch):
    """Sur Python < 3.11, tomllib n'existe pas : on le dit au lieu de planter."""
    ecrire_toml(tmp_path, '[voicechat]\nvoice = "af_heart"\n', monkeypatch)
    import builtins

    vrai_import = builtins.__import__

    def faux_import(nom, *args, **kwargs):
        if nom == "tomllib":
            raise ImportError("pas de tomllib")
        return vrai_import(nom, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", faux_import)
    reglages, avertissement = lire_config()
    assert reglages == {}
    assert avertissement is not None and "3.11" in avertissement


# ------------------------------------------------- réglages par modèle (v1.0)
TOML_MODELES = """\
[voicechat]
voice = "ff_siwis"

[modeles.gros-modele]
voice = "ff_siwis:3+ef_dora:1"
temperature = 0.2

[modeles."/mnt/data/models/autre.gguf"]
speed = 1.4
"""


def test_pour_modele_par_nom_court(tmp_path, monkeypatch):
    ecrire_toml(tmp_path, TOML_MODELES, monkeypatch)
    cfg, ajustes = Config().pour_modele("gros-modele.gguf")
    assert cfg.voice == "ff_siwis:3+ef_dora:1"
    assert cfg.temperature == pytest.approx(0.2)
    assert ajustes["voice"] == "ff_siwis:3+ef_dora:1"


def test_pour_modele_par_chemin_complet(tmp_path, monkeypatch):
    ecrire_toml(tmp_path, TOML_MODELES, monkeypatch)
    cfg, ajustes = Config().pour_modele("/mnt/data/models/autre.gguf")
    assert cfg.speed == pytest.approx(1.4)
    assert ajustes == {"speed": 1.4}


def test_pour_modele_sans_section_rend_la_meme_config(tmp_path, monkeypatch):
    ecrire_toml(tmp_path, TOML_MODELES, monkeypatch)
    cfg = Config(voice="if_sara")
    ajustee, ajustes = cfg.pour_modele("modele-inconnu.gguf")
    assert ajustes == {}
    assert ajustee.voice == "if_sara"


def test_pour_modele_ignore_les_cles_inconnues(tmp_path, monkeypatch):
    """Une clé inconnue dans une section de modèle est ignorée, pas transmise."""
    ecrire_toml(
        tmp_path,
        '[voicechat]\nvoice = "ff_siwis"\n\n[modeles.m]\nvoix = "x"\nvoice = "af_heart"\n',
        monkeypatch,
    )
    cfg, ajustes = Config().pour_modele("m")
    assert ajustes == {"voice": "af_heart"}
    assert cfg.voice == "af_heart"


def test_pour_modele_sans_nom(tmp_path, monkeypatch):
    ecrire_toml(tmp_path, TOML_MODELES, monkeypatch)
    cfg, ajustes = Config().pour_modele("")
    assert ajustes == {} and cfg.voice == "ff_siwis"
