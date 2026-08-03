"""Concurrency regression tests for the 2026-06-10 review finding:
repository.runtimes and gpu_allocator._allocations were read/mutated from
the heartbeat loop, the worker loop, and FastAPI sync routes with no lock —
a reconcile mutating the dict mid-heartbeat-iteration raised
"dictionary changed size during iteration", and racing allocates could
hand the same GPU to two deployments.

These hammer the shared state from threads; pre-fix they fail
probabilistically (reliably within the iteration counts used here).
"""
import threading

from greencompute_node_agent.domain.gpu_allocator import GpuAllocationError, GpuAllocator
from greencompute_node_agent.infrastructure.repository import NodeAgentRepository
from greencompute_protocol import UnifiedRuntimeRecord, WorkloadKind


def _runtime(i: int) -> UnifiedRuntimeRecord:
    return UnifiedRuntimeRecord(
        deployment_id=f"dep-{i}",
        workload_id=f"wl-{i}",
        hotkey="hk",
        node_id="n1",
        workload_kind=WorkloadKind.POD,
        status="ready",
    )


def _run_threads(workers: list) -> list[Exception]:
    errors: list[Exception] = []

    def _wrap(fn):
        def inner():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        return inner

    threads = [threading.Thread(target=_wrap(fn)) for fn in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_allocator_concurrent_allocate_release_stays_consistent():
    alloc = GpuAllocator(total_gpus=8)

    def churn(worker_id: int):
        def run():
            for _ in range(300):
                dep = f"dep-{worker_id}"
                try:
                    devices = alloc.allocate(dep, 2)
                    assert len(set(devices)) == 2
                finally:
                    alloc.release(dep)
        return run

    def observe():
        for _ in range(300):
            status = alloc.status()
            # Invariant: never more devices handed out than exist.
            assert status["used_gpus"] + status["free_gpus"] == 8
            flat = [d for ds in status["allocations"].values() for d in ds]
            assert len(flat) == len(set(flat)), f"double-allocated: {status}"

    errors = _run_threads([churn(i) for i in range(4)] + [observe, observe])
    errors = [e for e in errors if not isinstance(e, GpuAllocationError)]
    assert not errors, errors
    assert alloc.free_count == 8


def test_repository_mutation_during_iteration_does_not_explode(tmp_path):
    repo = NodeAgentRepository(state_path=str(tmp_path / "state.json"))

    def mutate():
        for i in range(400):
            repo.upsert_runtime(_runtime(i % 20))
            if i % 3 == 0:
                repo.remove_runtime(f"dep-{i % 20}")

    def iterate():
        for _ in range(400):
            repo.runtime_summary()
            repo.snapshot_runtimes()

    errors = _run_threads([mutate, mutate, iterate, iterate])
    assert not errors, errors  # pre-fix: RuntimeError(dict changed size)


def test_rehydrate_warns_but_records_overlap(caplog):
    alloc = GpuAllocator(total_gpus=4)
    alloc.rehydrate("dep-a", {0, 1})
    with caplog.at_level("WARNING"):
        alloc.rehydrate("dep-b", {1, 2})
    assert "overlaps" in caplog.text
    # Conservative: the overlap shrinks free capacity rather than hiding it.
    assert alloc.free_count == 1
    assert alloc.get_allocation("dep-b") == [1, 2]


# --- the inference proxy must not block the event loop (2026-08-03) ----------

# Read the source as TEXT: this suite has no fastapi, and importing routes.py
# to assert on its shape would only prove the test env has the dependency.
import pathlib as _pathlib

_ROUTES = _pathlib.Path(__file__).resolve().parents[2] / (
    "services/node-agent/src/greencompute_node_agent/transport/routes.py"
)


def test_inference_proxy_does_not_block_the_event_loop():
    """`async def` + blocking urlopen froze the WHOLE agent for the duration of
    an upstream call.

    On a big model that is ~50s with no /healthz, no capacity reports, nothing.
    The gateway's pre-flight health probe then times out and it rejects every
    concurrent request with "no healthy deployment available" — a concurrency
    ceiling of ONE, independent of the engine's own max_num_seqs. Observed
    live: 2 of 3 concurrent requests rejected.
    """
    src = _ROUTES.read_text()
    proxy = src[src.index("async def inference_proxy"):]
    proxy = proxy[:proxy.index("@router.", 10)]
    assert "asyncio.to_thread" in proxy, "blocking urlopen must run off the event loop"
    for line in proxy.splitlines():
        stripped = line.strip()
        if stripped.startswith("resp = ") and "urlopen" in stripped:
            assert "to_thread" in stripped, f"blocking call on the loop: {stripped}"


def test_proxy_timeout_accommodates_a_slow_reasoning_model():
    """~6 tok/s means a 600-token answer takes ~100s; the old 120s ceiling
    surfaced as a bare 502 mid-generation."""
    import re
    src = _ROUTES.read_text()
    m = re.search(r'GREENCOMPUTE_INFERENCE_PROXY_TIMEOUT_SECONDS", "(\d+)"', src)
    assert m and int(m.group(1)) >= 300, "proxy timeout too tight for a reasoning model"
