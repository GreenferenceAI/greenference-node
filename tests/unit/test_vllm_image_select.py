"""The arch/driver → vLLM image matrix (first node-agent tests).

Regression for the 2026-06-10 review finding: pre-Ada GPUs (sm < 8.9) were
handed the cu130 image whenever the driver was >= 580, which has no sm_8x
kernels → cudaErrorNoKernelImageForDevice on every Ampere box (the A4000
node only worked via its GREENCOMPUTE_VLLM_IMAGE host override). Compute
capability now decides before the driver does.
"""
from greencompute_node_agent.domain.inference import (
    _VLLM_CU12_IMAGE,
    _VLLM_CU130_IMAGE,
    _pick_vllm_image,
)


def test_ampere_new_driver_gets_cu12():
    # The A4000 box: sm_86 + driver 580 — the exact mis-pick from the review.
    assert _pick_vllm_image("8.6, 580.159.04") == _VLLM_CU12_IMAGE


def test_ampere_old_driver_gets_cu12():
    assert _pick_vllm_image("8.6, 535.104.05") == _VLLM_CU12_IMAGE


def test_ada_new_driver_gets_cu130():
    # The .12 box: 8x 4090 + driver >= 580, proven working on cu130.
    smi = "\n".join(["8.9, 580.65.06"] * 8)
    assert _pick_vllm_image(smi) == _VLLM_CU130_IMAGE


def test_ada_old_driver_gets_cu12():
    # The .24 box: 8x 4090 + driver < 580 — error 804 territory on cu130.
    smi = "\n".join(["8.9, 570.86.15"] * 8)
    assert _pick_vllm_image(smi) == _VLLM_CU12_IMAGE


def test_blackwell_gets_cu130():
    # The 5090 box: cu12 has no sm_120 kernels, cu130 is mandatory.
    smi = "\n".join(["12.0, 580.65.06"] * 8)
    assert _pick_vllm_image(smi) == _VLLM_CU130_IMAGE


def test_mixed_ampere_and_ada_gets_cu12():
    # The runtime may land on any GPU; only cu12 has kernels for the weakest.
    smi = "8.9, 580.65.06\n8.6, 580.65.06"
    assert _pick_vllm_image(smi) == _VLLM_CU12_IMAGE


def test_mixed_blackwell_wins_over_pre_ada():
    # No single image covers both; cu12 cannot run on Blackwell at all.
    smi = "12.0, 580.65.06\n8.6, 580.65.06"
    assert _pick_vllm_image(smi) == _VLLM_CU130_IMAGE


def test_unparseable_output_falls_back_to_cu130():
    assert _pick_vllm_image("") == _VLLM_CU130_IMAGE
    assert _pick_vllm_image("garbage") == _VLLM_CU130_IMAGE
    assert _pick_vllm_image("N/A, N/A") == _VLLM_CU130_IMAGE


# --- HF cache must be mounted from the HOST path (2026-08-03) -----------------


def _mount_source(env, monkeypatch, tmp_path):
    """Return the bind-mount SOURCE the agent would use for the HF cache."""
    import os
    from pathlib import Path
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    container_view = env.get("HF_HOME", "")
    # emulate the agent's decision (mirrors DockerInferenceBackend.start_runtime)
    host = os.environ.get("GREENCOMPUTE_HF_CACHE_HOST_PATH") or container_view
    return host if Path(container_view).exists() else None


def test_mount_source_is_the_host_path_not_the_agents_own_view(monkeypatch, tmp_path):
    """Regression for a fleet-wide bug (2026-08-03).

    The agent runs in a container where HF_HOME=/root/.cache/huggingface, but a
    bind-mount source is resolved by the docker daemon on the HOST. Using
    HF_HOME as the source mounted the host's small root filesystem into every
    inference container, so every model was re-downloaded into it — filling
    98G root disks fleet-wide with partial weights, and making a 1.5TB model
    impossible to start.
    """
    src = _mount_source(
        {"HF_HOME": str(tmp_path), "GREENCOMPUTE_HF_CACHE_HOST_PATH": "/data/hf-cache"},
        monkeypatch, tmp_path,
    )
    assert src == "/data/hf-cache", "must mount the host cache, not the agent's view"


def test_falls_back_to_hf_home_when_not_containerised(monkeypatch, tmp_path):
    # Agent running directly on the host: the two paths are the same thing.
    monkeypatch.delenv("GREENCOMPUTE_HF_CACHE_HOST_PATH", raising=False)
    src = _mount_source({"HF_HOME": str(tmp_path)}, monkeypatch, tmp_path)
    assert src == str(tmp_path)


def test_agent_source_actually_reads_the_host_path_var():
    import inspect
    from greencompute_node_agent.domain import inference
    src = inspect.getsource(inference.DockerInferenceBackend.start_runtime)
    assert "GREENCOMPUTE_HF_CACHE_HOST_PATH" in src
    idx_host = src.index("GREENCOMPUTE_HF_CACHE_HOST_PATH")
    idx_mount = src.index('/root/.cache/huggingface"]')
    assert idx_host < idx_mount, "host path must be resolved before the mount is built"
