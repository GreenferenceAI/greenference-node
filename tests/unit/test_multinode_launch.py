"""Distributed-replica launch planning (phase 2 of the multi-node build).

One replica spans several nodes: rank 0 is the Ray head and runs the API
server; the rest join and donate GPUs. These tests pin the command shapes and
the guards, so the topology is verifiable without a cluster.
"""
from greencompute_node_agent.domain.multinode_launch import (
    DEFAULT_RAY_PORT,
    build_cluster_wait_command,
    build_docker_command,
    build_nic_pin_command,
    docker_ulimit_flags,
    set_shm_size,
    strip_parallelism_flags,
    strip_port_publish,
    build_distributed_vllm_flags,
    build_head_entrypoint,
    build_ray_start_command,
    build_worker_entrypoint,
    docker_network_flags,
    is_worker_runtime,
    parse_multi_node_params,
    validate_params,
    worker_health,
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
    # Host IPC rides along with host networking: vLLM's API-server <-> engine
    # handshake and Ray's plasma store are both shared-memory transports.
    assert docker_network_flags() == ["--network", "host", "--ipc=host"]


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


# --- worker liveness (workers serve no HTTP) ---------------------------------


def test_worker_runtime_is_detected_from_metadata():
    assert is_worker_runtime({"multi_node_role": "worker"}) is True
    assert is_worker_runtime({"multi_node_role": "head"}) is False
    # Ordinary single-node runtimes must never be mistaken for a worker.
    assert is_worker_runtime({}) is False
    assert is_worker_runtime(None) is False


def test_worker_health_follows_the_container_not_an_endpoint():
    # A worker has no HTTP server; probing one would fail forever and tear the
    # rank down, rebuilding the replica in a loop.
    up = worker_health(True, "docker-vllm-backend")
    assert up["healthy"] is True and up["status"] == "ok" and up["role"] == "worker"
    down = worker_health(False)
    assert down["healthy"] is False and down["status"] == "unhealthy"


def test_image_precedes_the_entrypoint_args():
    cmd = rewrite(parse_multi_node_params(payload()))
    assert cmd[cmd.index("vllm/vllm-openai:v0.19.1") + 1] == "-c"


# --- the distributed flags must reach the ACTUAL command ----------------------
# Regression: build_distributed_vllm_flags was unit-tested in isolation but had
# ZERO callers, so vLLM launched with pipeline_parallel_size=1 and tried to load
# the whole model onto one node -> OOM. Assert on the final command, not the helper.


def test_head_command_carries_pipeline_parallel_and_ray_backend():
    script = rewrite(parse_multi_node_params(payload()))[-1]
    assert "--pipeline-parallel-size 8" in script
    assert "--distributed-executor-backend ray" in script
    assert "--tensor-parallel-size 8" in script


def test_single_node_tp_flag_is_not_duplicated():
    # The ordinary launcher already appended --tensor-parallel-size; the topology
    # is authoritative, so exactly one must survive.
    cmd = build_docker_command(
        docker_flags=["docker", "run", "-d"], image="img",
        serve_argv=["--model", "m", "--tensor-parallel-size", "4"],
        params=parse_multi_node_params(payload()),
    )
    assert cmd[-1].count("--tensor-parallel-size") == 1
    assert "--tensor-parallel-size 8" in cmd[-1]  # topology wins, not the stale 4


def test_strip_parallelism_flags_keeps_everything_else():
    assert strip_parallelism_flags(
        ["--model", "m", "--tensor-parallel-size", "4", "--max-model-len", "8192",
         "--distributed-executor-backend", "mp"]
    ) == ["--model", "m", "--max-model-len", "8192"]


def test_worker_command_has_no_vllm_flags_at_all():
    script = rewrite(parse_multi_node_params(payload(role="worker", rank=1)))[-1]
    assert "--pipeline-parallel-size" not in script
    assert "--distributed-executor-backend" not in script


# --- worker join must be order-independent -------------------------------------


def test_worker_retries_the_join_until_the_head_is_up():
    """Ranks start in arbitrary order and the head needs minutes (image pull +
    model load) before its Ray GCS listens. A single `ray start --address` fails
    instantly against a missing head; the worker then exited, the reconciler read
    a dead rank and rebuilt, and head/worker thrashed forever."""
    script = build_worker_entrypoint(parse_multi_node_params(payload(role="worker", rank=1)))[2]
    assert "until" in script and "sleep" in script, "join must be retried, not one-shot"
    assert "10.0.0.1:6379" in script
    assert "exit 1" in script, "must eventually give up so the agent sees a real failure"


def test_worker_join_window_matches_the_head_cluster_wait():
    p = parse_multi_node_params(payload(role="worker", rank=1, cluster_wait_seconds=900))
    assert "900" in build_worker_entrypoint(p)[2]


# --- Ray must exist in the image before we call it -----------------------------


def test_ray_is_bootstrapped_before_use_on_both_roles():
    """vllm/vllm-openai:*-cu130-* (what Blackwell/5090 auto-selects) ships WITHOUT
    ray — not the CLI, not the package. Without a bootstrap the rank dies on a
    bare `sh: 1: ray: not found` (observed on the 5090 cluster 2026-07-31)."""
    for p in (payload(), payload(role="worker", rank=1)):
        script = rewrite(parse_multi_node_params(p))[-1]
        assert "pip install" in script and "ray[default]" in script
        # the guard must come BEFORE any ray invocation
        assert script.index("command -v ray") < script.index("ray start")


def test_ray_bootstrap_is_a_noop_when_already_present():
    # cu12 images bundle ray; the guard must not reinstall it every start.
    script = rewrite(parse_multi_node_params(payload()))[-1]
    assert "command -v ray >/dev/null 2>&1 ||" in script


def test_cluster_wait_uses_python3_not_python():
    """The vLLM cu130 image (ubuntu 24.04) has /usr/bin/python3 and NO `python`.
    Bare `python` broke the && chain right after Ray started and the head
    container exited with no useful error (5090 cluster, 2026-07-31)."""
    cmd = build_cluster_wait_command(parse_multi_node_params(payload()))
    assert cmd.startswith("python3 -c")


# --- Ray object store must not fill the root disk ------------------------------


def test_ray_object_store_is_capped():
    """Ray defaults its object store to ~30% of RAM. On a 512GB box that is
    ~150GB, which won't fit /dev/shm, so Ray spills to /tmp on the small root
    filesystem, fills it, and the raylet dies mid-startup (5090 cluster,
    2026-07-31). PP only ships activations, so a few GB is plenty."""
    for p in (payload(), payload(role="worker", rank=1)):
        assert "--object-store-memory=" in build_ray_start_command(parse_multi_node_params(p))


def test_distributed_containers_get_a_bigger_shm():
    cmd = rewrite(parse_multi_node_params(payload()))
    assert "--shm-size" in cmd
    assert cmd[cmd.index("--shm-size") + 1] == "32g"
    assert cmd.count("--shm-size") == 1, "must replace the 8g default, not duplicate it"


def test_set_shm_size_replaces_and_preserves_other_flags():
    got = set_shm_size(["docker", "run", "-d", "--shm-size", "8g", "--gpus", "all"])
    assert got.count("--shm-size") == 1
    assert "8g" not in got and "32g" in got
    assert got[:2] == ["docker", "run"]  # still a docker run invocation
    assert "-d" in got and got[-2:] == ["--gpus", "all"]  # other flags preserved


# --- fd limit + NCCL interface pinning (proven on the 5090 cluster) ------------


def test_containers_raise_the_fd_limit():
    """Container default is 1024 fds. On a 256-CPU box Ray exhausts it during
    vLLM startup and the raylet stops accepting connections — surfacing as the
    misleading "raylet is dead"."""
    cmd = rewrite(parse_multi_node_params(payload()))
    assert "--ulimit" in cmd
    assert cmd[cmd.index("--ulimit") + 1].startswith("nofile=")


def test_nccl_and_gloo_are_pinned_to_the_cluster_nic():
    """With host networking NCCL/Gloo otherwise enumerate docker0 and every br-*
    bridge (172.x, not routable between hosts) and die with "unhandled system
    error". Gloo also rejects NCCL's ^exclude syntax, so a real NAME is needed."""
    snippet = build_nic_pin_command("172.16.106.12")
    assert "NCCL_SOCKET_IFNAME=$GC_NIC" in snippet
    assert "GLOO_SOCKET_IFNAME=$GC_NIC" in snippet
    assert "NCCL_IB_DISABLE=1" in snippet


def test_nic_probe_is_base64_encoded():
    # The probe is multi-line Python travelling through `docker run -> sh -c`;
    # inlining it got mangled by a layer of quoting every time.
    assert "base64 -d" in build_nic_pin_command("10.0.0.1")


def test_both_roles_pin_the_nic_before_starting_ray():
    for p in (payload(), payload(role="worker", rank=1)):
        script = rewrite(parse_multi_node_params(p))[-1]
        assert "GC_NIC" in script
        assert script.index("GC_NIC") < script.index("ray start")


# --- shell quoting + per-model engine passthrough (2026-08-01) ----------------


def test_head_entrypoint_shell_quotes_its_argv():
    """The head's argv is embedded in a `sh -c` script, so an argument
    containing a space MUST survive as one argument.

    This was already corrupting vision models before any operator input
    existed: `--limit-mm-per-prompt '{"image": 4}'` was naively joined and the
    shell re-split it into `{"image":` and `4}`.
    """
    from greencompute_node_agent.domain.multinode_launch import (
        MultiNodeParams, build_head_entrypoint,
    )
    p = MultiNodeParams(
        replica_id="r", role="head", rank=0, head_host="10.0.0.1",
        tensor_parallel_size=8, pipeline_parallel_size=2,
        node_count=2, gpus_per_node=8,
    )
    argv = ["--model", "m", "--limit-mm-per-prompt", '{"image": 4}']
    script = build_head_entrypoint(p, argv)[-1]
    assert "'{\"image\": 4}'" in script or '"{\\"image\\": 4}"' in script, script
    # and shlex must be able to round-trip the serve segment back to the argv
    import shlex
    assert '{"image": 4}' in shlex.split(script.split("&&")[-1])


def test_head_entrypoint_neutralises_injection_in_argv():
    from greencompute_node_agent.domain.multinode_launch import (
        MultiNodeParams, build_head_entrypoint,
    )
    p = MultiNodeParams(
        replica_id="r", role="head", rank=0, head_host="10.0.0.1",
        tensor_parallel_size=8, pipeline_parallel_size=2,
        node_count=2, gpus_per_node=8,
    )
    script = build_head_entrypoint(p, ["--model", "m; touch /tmp/pwned"])[-1]
    # the semicolon must be inside quotes, not a command separator
    assert "; touch /tmp/pwned" not in script.replace("'", "")[len("x"):] or "'" in script
    import shlex
    tail = shlex.split(script.split("&&")[-1])
    assert "m; touch /tmp/pwned" in tail, "must arrive as ONE argument"


def test_entry_command_is_not_shell_quoted():
    """`vllm_entry` is a shell FRAGMENT ("python3 -m vllm..."), not an argv
    element. Quoting it makes sh look for a binary literally named
    "python3 -m vllm.entrypoints.openai.api_server" and the head dies with
    'not found' — while the arguments after it must still be quoted."""
    from greencompute_node_agent.domain.multinode_launch import (
        MultiNodeParams, build_docker_command, DISTRIBUTED_VLLM_ENTRY,
    )
    p = MultiNodeParams(
        role="head", rank=0, head_host="10.0.0.1",
        tensor_parallel_size=8, pipeline_parallel_size=8,
        gpus_per_node=8, node_count=8, replica_id="r",
    )
    cmd = build_docker_command(
        docker_flags=["docker", "run", "-d"], image="img",
        serve_argv=["--model", "m", "--limit-mm-per-prompt", '{"image": 4}'],
        params=p,
    )
    serve_line = cmd[-1].split("&&")[-1]
    assert DISTRIBUTED_VLLM_ENTRY in serve_line, "entry must appear unquoted"
    assert f"'{DISTRIBUTED_VLLM_ENTRY}'" not in serve_line, "entry must NOT be quoted"
    # ...while a value containing a space is still protected
    import shlex
    assert '{"image": 4}' in shlex.split(serve_line)


# --- vLLM must advertise the cluster NIC too (2026-08-03) ---------------------


def test_nic_pin_exports_vllm_host_ip_not_just_nccl():
    """Pinning NCCL/Gloo is NOT enough.

    vLLM chooses its own advertised address independently of
    NCCL_SOCKET_IFNAME. With --network host these boxes expose docker0 and
    several br-* bridges, so unset it can bind an address peers cannot reach:
    the engine-core handshake fails with a bare "Engine core initialization
    failed" while NCCL looks perfectly healthy. Manual runs that set
    VLLM_HOST_IP worked; the agent path, which did not, failed every time.
    """
    from greencompute_node_agent.domain.multinode_launch import build_nic_pin_command
    cmd = build_nic_pin_command("172.16.106.13")
    assert "VLLM_HOST_IP" in cmd, "vLLM would pick an arbitrary interface"
    assert "NCCL_SOCKET_IFNAME" in cmd and "GLOO_SOCKET_IFNAME" in cmd


def test_nic_probe_emits_both_interface_and_address():
    import base64, re
    from greencompute_node_agent.domain.multinode_launch import build_nic_pin_command
    cmd = build_nic_pin_command("172.16.106.13")
    blob = re.search(r"echo ([A-Za-z0-9+/=]+) \| base64 -d", cmd).group(1)
    probe = base64.b64decode(blob).decode()
    # the probe must carry the ADDRESS through, not just the interface name
    assert 'r=i+" "+a' in probe, "probe discards the address VLLM_HOST_IP needs"
    assert '172.16.106.' in probe


def test_distributed_containers_get_host_ipc():
    """Shared-memory transports (vLLM handshake, Ray plasma) need the host IPC
    namespace; --shm-size alone does not substitute for it."""
    from greencompute_node_agent.domain.multinode_launch import docker_network_flags
    assert "--ipc=host" in docker_network_flags()


# --- host networking makes the container port the HOST port (2026-08-03) ------


def test_serve_port_is_retargeted_to_the_probed_host_port():
    """Latent bug: NO agent-managed distributed replica could ever go healthy.

    Single-node publishes `-p <ephemeral>:8000` so vLLM can hardcode 8000.
    The multi-node rewrite strips that mapping and adds --network host, so the
    container port IS the host port — but the serve args still said 8000 while
    runtime_url pointed at the ephemeral port. The head loaded perfectly, never
    answered on the probed port, and was torn down at the health deadline and
    rebuilt forever.
    """
    from greencompute_node_agent.domain.multinode_launch import set_serve_port
    assert set_serve_port(["--model", "m", "--host", "0.0.0.0", "--port", "8000"], 41234) == [
        "--model", "m", "--host", "0.0.0.0", "--port", "41234",
    ]


def test_serve_port_handles_equals_form_and_absence():
    from greencompute_node_agent.domain.multinode_launch import set_serve_port
    assert set_serve_port(["--port=8000"], 555) == ["--port=555"]
    assert set_serve_port(["--model", "m"], 777) == ["--model", "m", "--port", "777"]


def test_full_head_command_serves_on_the_host_port():
    from greencompute_node_agent.domain.multinode_launch import (
        MultiNodeParams, build_docker_command, set_serve_port,
    )
    p = MultiNodeParams(role="head", rank=0, head_host="10.0.0.1",
                        tensor_parallel_size=8, pipeline_parallel_size=8,
                        gpus_per_node=8, node_count=8, replica_id="r")
    cmd = build_docker_command(
        docker_flags=["docker", "run", "-d", "-p", "127.0.0.1:41234:8000"],
        image="img",
        serve_argv=set_serve_port(["--model", "m", "--port", "8000"], 41234),
        params=p,
    )
    script = cmd[-1]
    assert "--port 41234" in script, "head must listen where the agent probes"
    assert "-p" not in cmd[:cmd.index("img")], "host networking forbids -p"
    assert "--network" in cmd and "host" in cmd
