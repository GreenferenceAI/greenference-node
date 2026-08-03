"""Liveness verdicts for runtimes the agent already brought up.

The agent's reconcile loop used to `continue` past any runtime already in
`ready`, so a container that died afterwards stayed `ready` FOREVER. Nothing
re-checked it, nothing reported it, and for a distributed replica the validator
therefore saw every rank `ready`, chose action=keep, and never rebuilt — one
node fault turned into a permanent outage that only a human could spot.

The hard part is not detecting death, it is not *inventing* it: a false
positive tears down a healthy replica, which is worse than the bug. Hence two
different standards of proof:

  * the container is gone  -> DEAD immediately. There is no way to serve
    without a container, so this cannot be a false positive.
  * the container runs but the endpoint does not answer -> SUSPECT, and only
    DEAD after several consecutive failures. A single probe can fail for
    reasons that have nothing to do with the model (a busy box, a GC pause, a
    5s timeout landing badly).
"""
from __future__ import annotations

ALIVE = "alive"
SUSPECT = "suspect"
DEAD = "dead"

# Consecutive endpoint failures before a *running* container is declared dead.
# Reconcile runs on a timer, so this is roughly threshold x cycle-time of grace.
LIVENESS_FAILURE_THRESHOLD = 3


def rank_liveness(
    *,
    container_running: bool,
    endpoint_healthy: bool | None,
    consecutive_failures: int,
    threshold: int = LIVENESS_FAILURE_THRESHOLD,
) -> str:
    """Verdict for one already-running runtime.

    `endpoint_healthy` is None when the runtime has no endpoint to probe (a
    worker rank serves no HTTP by design) — such a rank is judged purely on its
    container.

    `consecutive_failures` is the count BEFORE this observation.
    """
    if not container_running:
        return DEAD
    if endpoint_healthy is None or endpoint_healthy:
        return ALIVE
    return DEAD if consecutive_failures + 1 >= max(1, threshold) else SUSPECT


def next_failure_count(verdict: str, consecutive_failures: int) -> int:
    """Failure counter to carry into the next cycle."""
    return 0 if verdict == ALIVE else consecutive_failures + 1
