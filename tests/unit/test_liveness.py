"""Liveness verdicts for runtimes already in `ready`.

Regression for the outage class found on 2026-08-03: the reconcile loop
`continue`d past every `ready` runtime, so a container that died afterwards
stayed `ready` forever. For a distributed replica that meant the validator saw
all ranks ready, chose action=keep, and never rebuilt — a single node fault
became a permanent outage (K3 sat dead for hours reporting healthy).
"""
from greencompute_node_agent.domain.liveness import (
    ALIVE,
    DEAD,
    SUSPECT,
    LIVENESS_FAILURE_THRESHOLD,
    next_failure_count,
    rank_liveness,
)


def test_a_missing_container_is_dead_immediately():
    """No container means no service — this cannot be a false positive, so it
    needs no grace period."""
    assert rank_liveness(
        container_running=False, endpoint_healthy=None, consecutive_failures=0
    ) == DEAD


def test_a_healthy_runtime_is_alive():
    assert rank_liveness(
        container_running=True, endpoint_healthy=True, consecutive_failures=0
    ) == ALIVE


def test_worker_rank_with_no_endpoint_is_judged_on_its_container_alone():
    # A worker serves no HTTP by design; endpoint_healthy is None.
    assert rank_liveness(
        container_running=True, endpoint_healthy=None, consecutive_failures=0
    ) == ALIVE
    assert rank_liveness(
        container_running=False, endpoint_healthy=None, consecutive_failures=0
    ) == DEAD


def test_one_missed_probe_does_not_kill_a_running_container():
    """The dangerous failure mode is INVENTING death: tearing down a healthy
    72-rank replica because one 5s probe landed badly is worse than the bug."""
    assert rank_liveness(
        container_running=True, endpoint_healthy=False, consecutive_failures=0
    ) == SUSPECT


def test_repeated_misses_eventually_declare_death():
    v = rank_liveness(
        container_running=True,
        endpoint_healthy=False,
        consecutive_failures=LIVENESS_FAILURE_THRESHOLD - 1,
    )
    assert v == DEAD


def test_the_full_suspect_to_dead_progression():
    failures = 0
    seen = []
    for _ in range(LIVENESS_FAILURE_THRESHOLD):
        v = rank_liveness(
            container_running=True, endpoint_healthy=False, consecutive_failures=failures
        )
        seen.append(v)
        failures = next_failure_count(v, failures)
    assert seen[:-1] == [SUSPECT] * (LIVENESS_FAILURE_THRESHOLD - 1)
    assert seen[-1] == DEAD


def test_recovery_resets_the_counter():
    """A blip followed by recovery must not accumulate toward a later kill."""
    failures = next_failure_count(SUSPECT, 0)
    assert failures == 1
    failures = next_failure_count(ALIVE, failures)
    assert failures == 0
    # so the next miss starts from scratch
    assert rank_liveness(
        container_running=True, endpoint_healthy=False, consecutive_failures=failures
    ) == SUSPECT


def test_threshold_of_one_still_requires_an_actual_failure():
    assert rank_liveness(
        container_running=True, endpoint_healthy=True, consecutive_failures=0, threshold=1
    ) == ALIVE
    assert rank_liveness(
        container_running=True, endpoint_healthy=False, consecutive_failures=0, threshold=1
    ) == DEAD


def test_zero_or_negative_threshold_cannot_disable_the_guard():
    # A misconfigured threshold must not make every probe fatal without proof.
    assert rank_liveness(
        container_running=True, endpoint_healthy=False, consecutive_failures=0, threshold=0
    ) == DEAD  # clamped to 1 — still requires a real failure, never kills a healthy one
    assert rank_liveness(
        container_running=True, endpoint_healthy=True, consecutive_failures=0, threshold=0
    ) == ALIVE


def test_reconcile_loop_actually_calls_the_liveness_check():
    """The logic is useless if `ready` runtimes still short-circuit."""
    import inspect
    from greencompute_node_agent.application import services
    src = inspect.getsource(services.NodeAgentService._reconcile_once_locked)
    assert "_check_runtime_liveness" in src, "ready runtimes are still skipped forever"


def test_dead_runtime_is_reported_so_the_validator_can_rebuild():
    import inspect
    from greencompute_node_agent.application import services
    src = inspect.getsource(services.NodeAgentService._check_runtime_liveness)
    assert "DeploymentState.FAILED" in src, "death must be reported, not just logged"
    assert "_terminate_runtime" in src, "a dead runtime must release its GPUs"


def test_liveness_is_scoped_to_inference_not_pods_or_vms():
    """Reaping a tenant's stopped pod would be far worse than the bug."""
    import inspect
    from greencompute_node_agent.application import services
    src = inspect.getsource(services.NodeAgentService._check_runtime_liveness)
    assert "WorkloadKind.INFERENCE" in src
