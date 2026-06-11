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
