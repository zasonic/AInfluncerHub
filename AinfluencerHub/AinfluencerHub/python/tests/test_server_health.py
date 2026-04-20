"""Smoke-tests the FastAPI endpoints that don't touch the GPU."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Run against a disposable settings file so tests don't clobber the user's config.
    monkeypatch.setenv("HOME", str(tmp_path))
    import importlib

    import server

    importlib.reload(server)
    return TestClient(server.app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_settings_endpoint_returns_dict(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    data = r.json()
    # Sanity-check: schema fields are present with correct types.
    assert isinstance(data.get("training_steps"), int)
    assert isinstance(data.get("hf_token"), str)


def test_settings_update_rejects_unknown_keys(client):
    r = client.put("/api/settings", json={"nonsense_key": 42, "hf_token": "hf_abc"})
    assert r.status_code == 200
    r2 = client.get("/api/settings")
    assert "nonsense_key" not in r2.json()
    assert r2.json()["hf_token"] == "hf_abc"
