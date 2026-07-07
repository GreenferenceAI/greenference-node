"""Theft-aware GPU allocation: a GPU busy per nvidia-smi but NOT in our own
allocation set belongs to a co-tenant/squatter, so the node must neither
advertise nor allocate it. Fixes the "theft-blind" over-commit that put a
customer onto the golden-miner's cards on .24.
"""
from greencompute_node_agent.domain.gpu_allocator import GpuAllocationError, GpuAllocator
from greencompute_node_agent.domain.gpu_probe import parse_busy_devices


def test_parse_busy_devices_threshold():
    # idle cards report ~1 MiB; busy ones hundreds of MiB to tens of GiB.
    csv = "0, 1\n1, 20238\n2, 3\n3, 600"
    assert parse_busy_devices(csv) == {1, 3}


def test_parse_busy_devices_ignores_garbage():
    assert parse_busy_devices("") == set()
    assert parse_busy_devices("N/A, N/A\nfoo") == set()


def test_allocate_skips_foreign_busy_devices():
    alloc = GpuAllocator(total_gpus=8)
    # A squatter holds GPUs 1-7; only GPU 0 is truly free.
    squatter = {1, 2, 3, 4, 5, 6, 7}
    got = alloc.allocate("dep-1", 1, avoid=squatter)
    assert got == [0]  # never lands on a stolen card


def test_allocate_raises_when_only_foreign_free():
    alloc = GpuAllocator(total_gpus=8)
    alloc.allocate("dep-0", 1)  # takes GPU 0
    squatter = {1, 2, 3, 4, 5, 6, 7}
    try:
        alloc.allocate("dep-1", 1, avoid=squatter)
        assert False, "should have raised — no usable free GPU"
    except GpuAllocationError as exc:
        assert "non-GreenCompute" in str(exc)


def test_allocated_devices_reports_our_usage():
    alloc = GpuAllocator(total_gpus=4)
    alloc.allocate("a", 2)
    alloc.allocate("b", 1)
    assert alloc.allocated_devices() == {0, 1, 2}
    alloc.release("a")
    assert alloc.allocated_devices() == {2}
