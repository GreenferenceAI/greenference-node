"""Distributed-replica launch planning (phase 2 of the multi-node build).

One replica spans several nodes: rank 0 is the Ray head and runs the API
server; the rest join and donate GPUs. These tests pin the command shapes and
the guards, so the topology is verifiable without a cluster.
"""
from greencompute_node_agent.domain.multinode_launch import (
    DEFAULT_RAY_PORT,
    build_cluster_wait_command,
    build_docker_command,
    strip_port_publish,
    build_distributed_vllm_flags,
    build_head_entrypoint,
    build_ray_start_command,
    build_worker_entrypoint,
    docker_network_flags,
    parse_multi_node_params,
    validate_params,
)


def payload(**over):
    mn = {
        "role": "head",
        "rank": 0,
        "head_host": "10.0.0.1",
        "node_count": 8,
        "gpus_per_node": 8,
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 8,
        "replica_id": "kimi-k3-r1",
    }
    mn.update(over)
    return {"image": "vllm/vllm-openai", "multi_node": mn}


# --- parsing -----------------------------------------------------------------


def test_single_node_payload_is_not_distributed():
    # The existing single-node path must stay completely untouched.
    assert parse_multi_node_params({"image": "x"}) is None
    assert parse_multi_node_params(None) is None
    assert parse_multi_node_params({"multi_node": "nonsense"}) is None


def test_parses_a_head_payload():
    p = parse_multi_node_params(payload())
    assert p.is_head and p.rank == 0
    assert p.world_size == 64 and p.total_gpus == 64
    assert p.head_port == DEFAULT_RAY_PORT


def test_parses_a_worker_payload():
    p = parse_multi_node_params(payload(role="worker", rank=3))
    assert not p.is_head and p.rank == 3


def test_parallelism_defaults_to_the_standard_topology():
    # TP within a node, PP across nodes — omit both and they derive.
    p = parse_multi_node_params(
        payload(tensor_parallel_size=None, pipeline_parallel_size=None)
    )
    assert p.tensor_parallel_size == 8  # gpus_per_node
    assert p.pipeline_parallel_size == 8  # node_count


def test_malformed_payload_returns_none_not_raises():
    for bad in [
        payload(role="captain"),
        payload(node_count="many"),
        {"multi_node": {}},
    ]:
        assert parse_multi_node_params(bad) is None


# --- validation --------------------------------------------------------------


def test_coherent_topology_passes():
    assert validate_params(parse_multi_node_params(payload())) == []


def test_world_size_must_match_reserved_gpus():
    # TP*PP = 4*8 = 32 but 8 nodes x 8 GPUs = 64 reserved — the mismatch that
    # otherwise surfaces as an opaque Ray placement-group failure.
    problems = validate_params(parse_multi_node_params(payload(tensor_parallel_size=4)))
    assert any("world size" in p for p in problems)


def test_tensor_parallel_may_not_span_nodes():
    p = parse_multi_node_params(payload(gpus_per_node=4, tensor_parallel_size=8,
                                        node_count=8, pipeline_parallel_size=4))
    assert any("cannot span nodes" in x for x in validate_params(p))


def test_worker_without_head_host_is_rejected():
    p = parse_multi_node_params(payload(role="worker", rank=1, head_host=""))
    assert any("no head_host" in x for x in validate_params(p))


def test_rank_zero_is_reserved_for_the_head():
    assert any("reserved for the head" in x
               for x in validate_params(parse_multi_node_params(payload(role="worker", rank=0))))


def test_head_must_be_rank_zero():
    assert any("head must be rank 0" in x
               for x in validate_params(parse_multi_node_params(payload(rank=2))))


def test_single_node_count_is_not_a_distributed_replica():
    p = parse_multi_node_params(payload(node_count=1, pipeline_parallel_size=1,
                                        tensor_parallel_size=8, gpus_per_node=8))
    assert any("node_count must be >= 2" in x for x in validate_params(p))


# --- command building --------------------------------------------------------


def test_head_starts_ray_head_on_its_port():
    cmd = build_ray_start_command(parse_multi_node_params(payload()))
    assert "--head" in cmd and f"--port={DEFAULT_RAY_PORT}" in cmd
    assert "--num-gpus=8" in cmd


def test_worker_dials_the_head():
    cmd = build_ray_start_command(parse_multi_node_params(payload(role="worker", rank=1)))
    assert f"--address=10.0.0.1:{DEFAULT_RAY_PORT}" in cmd
    assert "--head" not in cmd


def test_worker_entrypoint_blocks():
    argv = build_worker_entrypoint(parse_multi_node_params(payload(role="worker", rank=1)))
    assert argv[:2] == ["sh", "-c"]
    assert "--block" in argv[2]
    # A worker must never start an API server.
    assert "vllm serve" not in argv[2]


def test_head_entrypoint_waits_for_the_cluster_before_serving():
    p = parse_multi_node_params(payload())
    argv = build_head_entrypoint(p, ["vllm", "serve", "--model", "moonshot/kimi-k3"])
    script = argv[2]
    # Order matters: ray up, THEN wait for all GPUs, THEN serve. Serving early
    # is the classic race that kills the replica with "not enough resources".
    assert script.index("ray start") < script.index("cluster_resources")
    assert script.index("cluster_resources") < script.index("vllm serve")


def test_cluster_wait_targets_the_full_gpu_count():
    assert "expected = 64" in build_cluster_wait_command(parse_multi_node_params(payload()))


def test_distributed_flags_use_the_ray_backend():
    flags = build_distributed_vllm_flags(parse_multi_node_params(payload()))
    assert flags == [
        "--tensor-parallel-size", "8",
        "--pipeline-parallel-size", "8",
        "--distributed-executor-backend", "ray",
    ]


def test_distributed_replicas_use_host_networking():
    assert docker_network_flags() == ["--network", "host"]


# --- docker command rewrite --------------------------------------------------

DOCKER_FLAGS = [
    "docker", "run", "-d", "--name", "gc-kimi", "--shm-size", "8g",
    "-p", "127.0.0.1:8101:8000", "--gpus", "all",
]
SERVE_ARGV = ["--model", "moonshot/kimi-k3", "--host", "0.0.0.0", "--port", "8000"]


def rewrite(params):
    return build_docker_command(
        docker_flags=DOCKER_FLAGS, image="vllm/vllm-openai:v0.19.1",
        serve_argv=SERVE_ARGV, params=params,
    )


def test_port_publish_is_dropped_for_host_networking():
    # docker refuses `--network host` together with `-p`.
    cmd = rewrite(parse_multi_node_params(payload()))
    assert "-p" not in cmd and "127.0.0.1:8101:8000" not in cmd
    assert "--network" in cmd and "host" in cmd


def test_strip_port_publish_leaves_other_flags_intact():
    assert strip_port_publish(DOCKER_FLAGS) == [
        "docker", "run", "-d", "--name", "gc-kimi", "--shm-size", "8g", "--gpus", "all",
    ]


def test_head_command_serves_the_model():
    cmd = rewrite(parse_multi_node_params(payload()))
    assert cmd[-2] == "-c"
    script = cmd[-1]
    assert "ray start --head" in script
    assert "moonshot/kimi-k3" in script
    assert "--entrypoint" in cmd and "sh" in cmd


def test_worker_command_joins_and_serves_nothing():
    cmd = rewrite(parse_multi_node_params(payload(role="worker", rank=1)))
    script = cmd[-1]
    assert "--address=10.0.0.1" in script and "--block" in script
    assert "moonshot/kimi-k3" not in script  # workers never serve


def test_image_precedes_the_entrypoint_args():
    cmd = rewrite(parse_multi_node_params(payload()))
    assert cmd[cmd.index("vllm/vllm-openai:v0.19.1") + 1] == "-c"
