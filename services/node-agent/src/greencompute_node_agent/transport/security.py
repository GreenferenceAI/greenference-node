"""Auth validation for unified node agent endpoints."""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Warn-once flags so a misconfigured node logs its auth posture without spamming.
_warned = {"open": False, "closed": False}


def effective_agent_secret(cfg) -> str | None:
    """The node-agent's :8007 auth secret.

    Falls back across the three historical env names (agent/inference/compute)
    so an operator who set any one of them is honoured. The gateway sends
    GREENCOMPUTE_INFERENCE_AUTH_SECRET as X-Agent-Auth, so inference_auth_secret
    must be accepted. miner_auth_secret is DELIBERATELY excluded — it has a weak
    shared default ("greencompute-node-local-secret") and must never gate the API.
    """
    for attr in ("agent_auth_secret", "inference_auth_secret", "compute_auth_secret"):
        val = getattr(cfg, attr, None)
        if val:
            return val
    return None


def validate_auth(header: str | None, cfg, *, sensitive: bool) -> None:
    """Authenticate an X-Agent-Auth header against the configured node secret.

    - A secret IS configured → require a constant-time match (401 otherwise).
    - NO secret configured:
        * sensitive=True  (control/ops/SSH-key/terminate/register/...) → FAIL
          CLOSED with 503. These must never be world-open, so they are disabled
          until the operator sets the secret. Nothing on the gateway's hot path
          calls them, so this is safe to ship before the secret is rolled out.
        * sensitive=False (inference proxy / healthz / pod stats) → allowed, but
          logged once as UNAUTHENTICATED, so inference keeps working until the
          shared secret is rolled out to both the nodes and the gateway. After
          that it is enforced like everything else.
    """
    expected = effective_agent_secret(cfg)
    if expected is not None:
        if not header or not hmac.compare_digest(header, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or missing auth header",
            )
        return

    if sensitive:
        if not _warned["closed"]:
            logger.error(
                "node-agent auth secret is NOT set (GREENCOMPUTE_INFERENCE_AUTH_SECRET / "
                "AGENT / COMPUTE) — control routes (ssh/terminate/register/capacity/...) "
                "are DISABLED (fail-closed). Set the secret to enable them."
            )
            _warned["closed"] = True
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="node-agent auth not configured; control route disabled",
        )

    if not _warned["open"]:
        logger.warning(
            "node-agent auth secret is NOT set — inference/stats routes are "
            "UNAUTHENTICATED. Set GREENCOMPUTE_INFERENCE_AUTH_SECRET on the nodes "
            "AND the gateway to require auth."
        )
        _warned["open"] = True


def validate_optional_auth(header: str | None, expected: str | None) -> None:
    """Back-compat shim (constant-time). Prefer validate_auth()."""
    if expected is None:
        return
    if not header or not hmac.compare_digest(header, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing auth header",
        )
