"""Tensor-parallel size must reach vLLM.

Regression for the 2026-07-13 finding: the allocator reserved N GPUs for an
inference runtime, but `tensor_parallel_size` was never put in the payload, so
`inference.py` read None and vLLM launched at its TP=1 default — the model
loaded onto ONE card while the other N-1 sat idle and reserved, and anything
that genuinely needed them OOMed. This blocked serving any large model on an
8x5090 node.
"""
from greencompute_node_agent.domain.inference import resolve_tensor_parallel_size


def test_derives_from_allocated_gpu_count():
    # The 8x5090 case — the whole point of the fix.
    assert resolve_tensor_parallel_size(8) == 8
    assert resolve_tensor_parallel_size(4) == 4
    assert resolve_tensor_parallel_size(2) == 2


def test_single_gpu_stays_one():
    assert resolve_tensor_parallel_size(1) == 1


def test_explicit_override_wins():
    # An operator pinning TP for a model whose head count doesn't divide by 8.
    assert resolve_tensor_parallel_size(8, 4) == 4
    assert resolve_tensor_parallel_size(8, "2") == 2


def test_junk_override_falls_back_to_gpu_count():
    for junk in (None, "", "abc", 0, -3, {}):
        assert resolve_tensor_parallel_size(8, junk) == 8


def test_junk_gpu_count_is_safe():
    for junk in (None, "abc", 0, -1):
        assert resolve_tensor_parallel_size(junk) == 1
