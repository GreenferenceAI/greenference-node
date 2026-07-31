"""Launching one node's share of a distributed (multi-node) inference replica.

A model too large for one chassis is served as ONE vLLM engine spanning several
nodes, coordinated by Ray. Each node runs the same image but plays a different
role:

  * **head** (rank 0) — starts the Ray head, waits for every worker's GPUs to
    join, then runs `vllm serve`. This is the only node with an API server, so
    it is the one the gateway routes inference to.
  * **worker** (rank > 0) — joins the head's Ray cluster and blocks, donating
    its GPUs. It serves no HTTP itself.

Everything here is pure string/command building so the topology can be tested
without a cluster. The validator's placement planner decides *which* nodes take
which rank (see the api repo's `domain/multinode.py`); this module turns that
decision into the commands one node actually runs.

Correctness notes that cost real debugging time if missed:
  * vLLM's world size is `tensor_parallel_size × pipeline_parallel_size`, and it
    must equal `node_count × gpus_per_node`. A mismatch fails minutes into
    startup with an opaque Ray placement-group error, so we check it up front.
  * The head must not start vLLM until every worker has joined. Ray accepts the
    head immediately, so a naive `ray start && vllm serve` races: vLLM sees only
    the head's GPUs and dies "not enough resources". Hence the wait loop.
  * Ray uses a spread of dynamic ports between nodes, so distributed replicas
    need host networking. That is a deliberate trade-off, acceptable only
    because these run on a dedicated co-located cluster — see `HOST_NETWORK_NOTE`.
"""
from __future__ import annotations

from dataclasses import dataclass

HEAD = "head"
WORKER = "worker"

# Ray's GCS port on the head. Workers dial this to join.
DEFAULT_RAY_PORT = 6379
# How long the head waits for workers before giving up, in seconds. Generous:
# pulling a multi-hundred-GB model onto several boxes is slow.
DEFAULT_CLUSTER_WAIT_SECONDS = 1800

HOST_NETWORK_NOTE = (
    "Ray allocates dynamic ports for object-manager/node-manager traffic between "
    "peers, so distributed replicas run with host networking. Only deploy these "
    "on a dedicated, firewalled cluster — the node's ports are exposed to its "
    "network segment."
)


@dataclass(frozen=True)
class MultiNodeParams:
    """This node's share of a distributed replica."""

    role: str  # "head" | "worker"
    rank: int
    head_host: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    gpus_per_node: int
    node_count: int
    replica_id: str = ""
    head_port: int = DEFAULT_RAY_PORT
    cluster_wait_seconds: int = DEFAULT_CLUSTER_WAIT_SECONDS

    @property
    def is_head(self) -> bool:
        return self.role == HEAD

    @property
    def world_size(self) -> int:
        """GPUs vLLM expects across the whole cluster."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def total_gpus(self) -> int:
        """GPUs the placement actually reserved."""
        return self.node_count * self.gpus_per_node


def parse_multi_node_params(payload: dict | None) -> MultiNodeParams | None:
    """Extract distributed-replica params from an inference payload.

    Returns None for an ordinary single-node deployment, so callers can keep the
    existing path untouched. Malformed input also returns None rather than
    raising — a broken distributed payload must not take down the agent; it
    surfaces as the runtime failing to start.
    """
    if not isinstance(payload, dict):
        return None
    raw = payload.get("multi_node")
    if not isinstance(raw, dict):
        return None
    try:
        role = str(raw.get("role", "")).lower()
        if role not in {HEAD, WORKER}:
            return None
        node_count = int(raw["node_count"])
        gpus_per_node = int(raw["gpus_per_node"])
        params = MultiNodeParams(
            role=role,
            rank=int(raw.get("rank", 0)),
            head_host=str(raw.get("head_host", "")).strip(),
            tensor_parallel_size=int(raw.get("tensor_parallel_size") or gpus_per_node),
            pipeline_parallel_size=int(raw.get("pipeline_parallel_size") or node_count),
            gpus_per_node=gpus_per_node,
            node_count=node_count,
            replica_id=str(raw.get("replica_id", "")),
            head_port=int(raw.get("head_port") or DEFAULT_RAY_PORT),
            cluster_wait_seconds=int(
                raw.get("cluster_wait_seconds") or DEFAULT_CLUSTER_WAIT_SECONDS
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    return params


def validate_params(params: MultiNodeParams) -> list[str]:
    """Reasons this node can't start its share, empty when coherent."""
    problems: list[str] = []
    if params.node_count < 2:
        problems.append("node_count must be >= 2 for a distributed replica")
    if params.gpus_per_node < 1:
        problems.append("gpus_per_node must be >= 1")
    if params.world_size != params.total_gpus:
        problems.append(
            f"world size {params.tensor_parallel_size}x{params.pipeline_parallel_size}"
            f"={params.world_size} != reserved GPUs {params.node_count}x{params.gpus_per_node}"
            f"={params.total_gpus}"
        )
    if params.tensor_parallel_size > params.gpus_per_node:
        problems.append(
            f"tensor_parallel_size {params.tensor_parallel_size} exceeds gpus_per_node "
            f"{params.gpus_per_node} — tensor parallelism cannot span nodes"
        )
    if not params.is_head and not params.head_host:
        problems.append("worker has no head_host to join")
    if params.is_head and params.rank != 0:
        problems.append(f"head must be rank 0, got {params.rank}")
    if not params.is_head and params.rank == 0:
        problems.append("rank 0 is reserved for the head")
    return problems


# Some vLLM images ship without Ray — notably vllm/vllm-openai:*-cu130-*, which
# is exactly the image Blackwell (5090, compute cap 12.0) auto-selects. The
# older cu12 image bundles it, so this only pays a cost where it's missing. The
# failure without this is a bare `sh: 1: ray: not found` and a dead rank.
# TODO: bake Ray into a GreenCompute vLLM image so distributed ranks don't
# pip-install on every container start (needs egress + ~40s).
RAY_BOOTSTRAP = 'command -v ray >/dev/null 2>&1 || pip install -q "ray[default]"'


def build_ray_start_command(params: MultiNodeParams) -> str:
    """The `ray start` invocation for this node's role."""
    if params.is_head:
        return (
            f"ray start --head --port={params.head_port} "
            f"--num-gpus={params.gpus_per_node} --disable-usage-stats"
        )
    return (
        f"ray start --address={params.head_host}:{params.head_port} "
        f"--num-gpus={params.gpus_per_node} --disable-usage-stats"
    )


def build_cluster_wait_command(params: MultiNodeParams) -> str:
    """Block until every worker's GPUs have joined the Ray cluster.

    Without this the head races ahead and vLLM dies claiming insufficient
    resources, because Ray reports the head's GPUs the instant it starts.
    """
    snippet = "\n".join([
        "import ray, sys, time",
        "ray.init(address='auto')",
        f"expected = {params.total_gpus}",
        f"deadline = time.time() + {params.cluster_wait_seconds}",
        "gpus = lambda: ray.cluster_resources().get('GPU', 0)",
        "while gpus() < expected and time.time() < deadline:",
        "    time.sleep(5)",
        "sys.exit(0 if gpus() >= expected else 1)",
    ])
    return f'python -c "{snippet}"'


def build_distributed_vllm_flags(params: MultiNodeParams) -> list[str]:
    """Extra `vllm serve` flags that turn a single-node launch into a Ray-backed
    distributed one. Appended to the normal command by the caller."""
    return [
        "--tensor-parallel-size", str(params.tensor_parallel_size),
        "--pipeline-parallel-size", str(params.pipeline_parallel_size),
        "--distributed-executor-backend", "ray",
    ]


def build_worker_entrypoint(params: MultiNodeParams) -> list[str]:
    """A worker joins the cluster and blocks forever donating its GPUs.

    RETRIES the join, because `ray start --address=...` fails immediately when
    the head's GCS isn't listening yet — and ranks start in arbitrary order: the
    head has to pull its image and load a multi-GB model before its Ray is even
    up. Failing fast made the worker exit, which the reconciler read as a dead
    rank and rebuilt the replica, so head and worker thrashed and neither ever
    came up (observed on the fleet 2026-07-30). Retrying over the same window
    the head uses for its cluster-wait makes bring-up order-independent.

    `--block` then keeps PID 1 alive; when the head tears down, the worker's Ray
    session ends and the container exits.
    """
    join = build_ray_start_command(params)
    # Poll until the head accepts us or the window closes; exit non-zero on
    # timeout so the agent reports a real failure rather than hanging forever.
    script = (
        f"{RAY_BOOTSTRAP}; "
        f"deadline=$(( $(date +%s) + {params.cluster_wait_seconds} )); "
        f'until {join} --block; do '
        f'  if [ $(date +%s) -ge $deadline ]; then '
        f'    echo "multi-node worker: head {params.head_host}:{params.head_port} '
        f'unreachable after {params.cluster_wait_seconds}s" >&2; exit 1; '
        f"  fi; "
        f'  echo "waiting for ray head {params.head_host}:{params.head_port}..."; '
        f"  sleep 10; "
        f"done"
    )
    return ["sh", "-c", script]


def build_head_entrypoint(params: MultiNodeParams, vllm_argv: list[str]) -> list[str]:
    """Head: bring up Ray, wait for the full cluster, then serve.

    `vllm_argv` is the ordinary single-node vLLM command; the distributed flags
    are expected to already be part of it (see build_distributed_vllm_flags).
    """
    serve = " ".join(vllm_argv)
    script = " && ".join([
        RAY_BOOTSTRAP,
        build_ray_start_command(params),
        build_cluster_wait_command(params),
        serve,
    ])
    return ["sh", "-c", script]


def docker_network_flags() -> list[str]:
    """Ray peers need host networking (see HOST_NETWORK_NOTE)."""
    return ["--network", "host"]


# How to invoke the vLLM API server once we've overridden the image entrypoint
# with `sh`. The module form works on both vLLM images we ship (0.8.5 cu12 and
# 0.19.1 cu130); `vllm serve` only exists on newer builds. Overridable per
# deployment for images that differ.
DISTRIBUTED_VLLM_ENTRY = "python3 -m vllm.entrypoints.openai.api_server"


WORKER_ROLE_KEY = "multi_node_role"


def is_worker_runtime(metadata: dict | None) -> bool:
    """Is this runtime a non-serving rank of a distributed replica?

    Workers run `ray start --block` and expose NO HTTP server, so every
    HTTP-based readiness/health path must skip them. Without this a worker fails
    its first health probe, gets torn down, and the replica rebuilds forever.
    """
    return bool(metadata) and metadata.get(WORKER_ROLE_KEY) == WORKER


def worker_health(container_running: bool, backend_name: str = "") -> dict:
    """Health verdict for a worker rank.

    A worker's liveness is 'is its container (and therefore its Ray session)
    still up', not 'does an endpoint answer' — it has no endpoint to answer.
    """
    return {
        "status": "ok" if container_running else "unhealthy",
        "healthy": bool(container_running),
        "backend": backend_name,
        "role": WORKER,
    }


def strip_port_publish(docker_flags: list[str]) -> list[str]:
    """Drop `-p host:port:container` pairs.

    Host networking and `-p` are mutually exclusive — docker refuses the run if
    both are given. The head's API port is reachable directly on the host.
    """
    out: list[str] = []
    skip_next = False
    for flag in docker_flags:
        if skip_next:
            skip_next = False
            continue
        if flag in ("-p", "--publish"):
            skip_next = True
            continue
        out.append(flag)
    return out


def strip_parallelism_flags(serve_argv: list[str]) -> list[str]:
    """Remove any parallelism flags the single-node path already added.

    The ordinary launcher appends `--tensor-parallel-size N` from the allocated
    GPU count. For a distributed replica the authoritative degrees come from the
    topology, so drop these and let build_distributed_vllm_flags set them —
    otherwise the flag appears twice and whichever vLLM picks last silently wins.
    """
    drop = {"--tensor-parallel-size", "--pipeline-parallel-size", "--distributed-executor-backend"}
    out: list[str] = []
    skip_next = False
    for arg in serve_argv:
        if skip_next:
            skip_next = False
            continue
        if arg in drop:
            skip_next = True
            continue
        out.append(arg)
    return out


def build_docker_command(
    *,
    docker_flags: list[str],
    image: str,
    serve_argv: list[str],
    params: MultiNodeParams,
    vllm_entry: str = DISTRIBUTED_VLLM_ENTRY,
) -> list[str]:
    """Rewrite a single-node `docker run <flags> <image> <serve args>` into this
    node's role in a distributed replica.

    The image entrypoint is replaced with `sh` so we can sequence Ray bring-up,
    the cluster wait, and the server in one container lifetime. The head's serve
    command gets the topology's parallelism flags — without them vLLM defaults to
    pipeline_parallel_size=1 and tries to load the WHOLE model onto this node's
    GPUs, which is exactly the OOM a distributed replica exists to avoid.
    """
    flags = strip_port_publish(docker_flags) + docker_network_flags() + ["--entrypoint", "sh"]
    if params.is_head:
        head_argv = [
            vllm_entry,
            *strip_parallelism_flags(serve_argv),
            *build_distributed_vllm_flags(params),
        ]
        argv = build_head_entrypoint(params, head_argv)
    else:
        argv = build_worker_entrypoint(params)
    # argv is ["sh", "-c", script]; the image supplies the "sh".
    return [*flags, image, *argv[1:]]
