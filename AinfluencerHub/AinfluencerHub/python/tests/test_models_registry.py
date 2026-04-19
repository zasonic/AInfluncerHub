"""Verify the central model registry stays consistent."""

from services import models


def test_all_specs_have_repo_id():
    for key, spec in models.ALL.items():
        assert spec.repo_id, f"{key} has empty repo_id"
        assert spec.purpose, f"{key} has empty purpose"
        assert spec.size_gb > 0, f"{key} has non-positive size"


def test_ip_adapter_has_subartifact_info():
    # IP-Adapter must point to a specific weight file — generation breaks otherwise.
    assert models.IP_ADAPTER.subfolder == "sdxl_models"
    assert models.IP_ADAPTER.weight_name and models.IP_ADAPTER.weight_name.endswith(
        ".safetensors"
    )


def test_get_unknown_key_raises():
    import pytest

    with pytest.raises(KeyError):
        models.get("not_a_real_model")


def test_manifest_shape():
    manifest = models.manifest()
    assert set(manifest) == set(models.ALL)
    for entry in manifest.values():
        assert {"hf_id", "size_gb", "purpose", "required", "revision"} <= entry.keys()
