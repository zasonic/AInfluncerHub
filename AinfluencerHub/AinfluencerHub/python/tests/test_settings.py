"""Settings schema round-trip and validation."""

import json

import pytest
from core.settings import DEFAULTS, Settings, SettingsModel


def test_defaults_match_model():
    assert DEFAULTS == SettingsModel().model_dump()


def test_round_trip(tmp_path):
    s = Settings(path=tmp_path / "settings.json")
    s.set("hf_token", "hf_abc123")
    s.set("training_steps", 1500)

    reloaded = Settings(path=tmp_path / "settings.json")
    assert reloaded.get("hf_token") == "hf_abc123"
    assert reloaded.get("training_steps") == 1500


def test_unknown_keys_are_dropped(tmp_path, caplog):
    s = Settings(path=tmp_path / "settings.json")
    with caplog.at_level("WARNING"):
        s.update({"definitely_not_a_field": "boom", "hf_token": "ok"})

    # Known key persisted.
    assert s.get("hf_token") == "ok"
    # Unknown key was not silently stored.
    on_disk = json.loads((tmp_path / "settings.json").read_text())
    assert "definitely_not_a_field" not in on_disk


def test_invalid_type_rejected(tmp_path):
    s = Settings(path=tmp_path / "settings.json")
    before = s.get("training_steps")
    s.update({"training_steps": "not a number"})
    # Rejected — previous value kept.
    assert s.get("training_steps") == before


def test_range_constraint(tmp_path):
    s = Settings(path=tmp_path / "settings.json")
    before = s.get("lora_rank")
    # Schema caps lora_rank at 256.
    s.update({"lora_rank": 9999})
    assert s.get("lora_rank") == before


def test_corrupted_file_falls_back_to_defaults(tmp_path, caplog):
    path = tmp_path / "settings.json"
    path.write_text("not valid json at all")
    with caplog.at_level("WARNING"):
        s = Settings(path=path)
    assert s.get("training_steps") == SettingsModel().training_steps


@pytest.mark.parametrize(
    "key,expected_type",
    [
        ("hf_token",       str),
        ("training_steps", int),
        ("lora_rank",      int),
        ("setup_complete", bool),
    ],
)
def test_types_are_enforced(tmp_path, key, expected_type):
    s = Settings(path=tmp_path / "settings.json")
    assert isinstance(s.get(key), expected_type)
