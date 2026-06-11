"""Observability routes must never leak SSH private keys or HF tokens."""

from __future__ import annotations

from greencompute_node_agent.transport.routes import _is_secret_key, _redact


def test_secret_key_classification():
    assert _is_secret_key("ssh_private_key")
    assert _is_secret_key("hf_token")
    assert _is_secret_key("HF_TOKEN")
    assert _is_secret_key("api_secret")
    assert _is_secret_key("password")
    assert _is_secret_key("token")
    # Non-secret metadata must stay visible.
    assert not _is_secret_key("ssh_public_keys")
    assert not _is_secret_key("ssh_fingerprint")
    assert not _is_secret_key("gpu_devices")
    assert not _is_secret_key("port_allocations")


def test_redact_strips_secrets_but_keeps_public_metadata():
    meta = {
        "image": "x",
        "ssh_public_keys": ["ssh-ed25519 AAAA..."],
        "ssh_fingerprint": "ab:cd",
        "ssh_private_key": "-----BEGIN-----secret-----END-----",
        "hf_token": "hf_supersecret",
        "gpu_devices": [0, 1],
        "port_allocations": {"8080": 12345},
        "nested": {"hf_token": "hf_inner", "api_secret": "zzz", "ssh_public_keys": ["pub"]},
    }
    out = _redact({"deployment_id": "d1", "metadata": meta})
    m = out["metadata"]
    assert m["ssh_private_key"] == "***redacted***"
    assert m["hf_token"] == "***redacted***"
    assert m["nested"]["hf_token"] == "***redacted***"
    assert m["nested"]["api_secret"] == "***redacted***"
    # Public material is preserved verbatim.
    assert m["ssh_public_keys"] == ["ssh-ed25519 AAAA..."]
    assert m["nested"]["ssh_public_keys"] == ["pub"]
    assert m["ssh_fingerprint"] == "ab:cd"
    assert m["gpu_devices"] == [0, 1]


def test_no_gpu_template_has_zero_gpu_fraction():
    # Guards the CPU-only-pod fix: ubuntu-ssh must signal no GPU.
    from greencompute_node_agent.domain.templates import get_template

    assert get_template("ubuntu-ssh").gpu_fraction == 0.0
    assert get_template("gpu-pod").gpu_fraction == 1.0
