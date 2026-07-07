"""wait_for_ready must verify sshd host-side (docker), not by dialing the
node's own public SSH host — which NAT-hairpins on our providers and made the
old check false-negative on EVERY pod (23 'SSH not reachable' warnings on .24,
zero successes), wasting the timeout and never actually confirming SSH.
"""
import subprocess
from types import SimpleNamespace

from greencompute_node_agent.domain.pod import ProcessPodBackend


def _fake_docker_top(stdout: str, returncode: int = 0):
    def _run(cmd, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return _run


def test_sshd_running_true_when_sshd_in_process_list(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        _fake_docker_top("UID PID CMD\nroot 12 /usr/sbin/sshd\n"),
    )
    assert ProcessPodBackend._sshd_running("c1") is True


def test_sshd_running_false_when_absent(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        _fake_docker_top("UID PID CMD\nroot 1 sleep infinity\n"),
    )
    assert ProcessPodBackend._sshd_running("c1") is False


def test_sshd_running_false_on_docker_error(monkeypatch):
    def _boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 5.0)
    monkeypatch.setattr(subprocess, "run", _boom)
    assert ProcessPodBackend._sshd_running("c1") is False


def test_wait_for_ready_no_container_short_circuits():
    backend = ProcessPodBackend()
    rt = SimpleNamespace(container_id=None, ssh_port=32000, ssh_host="1.2.3.4")
    assert backend.wait_for_ready(rt, timeout_seconds=1.0) is True


def test_wait_for_ready_returns_true_once_sshd_up(monkeypatch):
    backend = ProcessPodBackend()
    rt = SimpleNamespace(container_id="c1", ssh_port=32000, ssh_host="1.2.3.4")
    monkeypatch.setattr(
        ProcessPodBackend, "_sshd_running", staticmethod(lambda cid: True)
    )
    assert backend.wait_for_ready(rt, timeout_seconds=5.0) is True


def test_wait_for_ready_times_out_when_sshd_never_up(monkeypatch):
    backend = ProcessPodBackend()
    rt = SimpleNamespace(container_id="c1", ssh_port=32000, ssh_host="1.2.3.4")
    monkeypatch.setattr(
        ProcessPodBackend, "_sshd_running", staticmethod(lambda cid: False)
    )
    # A short timeout so the poll loop exits quickly.
    assert backend.wait_for_ready(rt, timeout_seconds=0.1) is False
