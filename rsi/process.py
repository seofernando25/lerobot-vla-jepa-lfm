"""Logged subprocesses and Linux filesystem isolation. No shell execution."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from rsi.core import atomic


class OperatorStop(InterruptedError):
    """Operator-requested stop; safe to resume later."""


class ProcessTimeout(TimeoutError):
    """External process exceeded its fixed wall-clock budget."""


def codex_argv():
    return [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "workspace-write",
        "-m",
        "gpt-6-astra",
        "-c",
        'model_reasoning_effort="low"',
        "-c",
        'approval_policy="never"',
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        "features.multi_agent=false",
        "--json",
        "-",
    ]


def isolated(argv, repo, workspace, output=None, evaluator=False):
    """Hide the repository (including all state/git), expose only the selected copy.

    System/dependency files are read-only; /tmp is private. Bubblewrap is mandatory,
    not a best-effort sandbox. Codex retains its own workspace sandbox as well.
    """
    command = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        str(repo.parent),
        "--ro-bind" if evaluator else "--bind",
        str(workspace),
        "/tmp/work",
        "--chdir",
        "/tmp/work",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONPATH",
        "/tmp/work/src",
    ]
    if output:
        command += ["--bind", str(output), "/tmp/output"]
    if evaluator:
        command += [
            "--setenv",
            "HF_DATASETS_CACHE",
            "/tmp/hf-datasets-cache",
            "--setenv",
            "HF_HUB_OFFLINE",
            "1",
            "--setenv",
            "TRANSFORMERS_OFFLINE",
            "1",
        ]
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        hub_cache = Path(
            os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub"))
        )
        if hub_cache.is_dir():
            command += [
                "--ro-bind",
                str(hub_cache.resolve()),
                "/tmp/hf-model-cache",
                "--setenv",
                "HF_HUB_CACHE",
                "/tmp/hf-model-cache",
                "--setenv",
                "HUGGINGFACE_HUB_CACHE",
                "/tmp/hf-model-cache",
            ]
        venv = repo / ".venv"
        raw_executable = Path(argv[0])
        if venv.exists() and str(raw_executable).startswith(str(venv)):
            command += ["--ro-bind", str(venv), "/tmp/venv"]
            argv = ["/tmp/venv/bin/python", *argv[1:]]
    if not evaluator:
        executable = shutil.which(argv[0])
        if executable is None:
            raise RuntimeError("Codex CLI not installed")
        # Fresh CODEX_HOME under private /tmp: no sessions, global instructions,
        # hooks, MCPs or skills. Only auth is mounted read-only.
        codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        command += ["--setenv", "CODEX_HOME", "/tmp/codex-home", "--dir", "/tmp/codex-home"]
        auth = codex_home / "auth.json"
        if auth.exists():
            command += ["--ro-bind", str(auth), "/tmp/codex-home/auth.json"]
        argv = [str(Path(executable).resolve()), *argv[1:]]
    return command + ["--", *argv]


def run_logged(argv, cwd, logs, timeout, stop_file, prompt=None):
    logs.mkdir(parents=True, exist_ok=True)
    atomic(logs / "command.json", {"argv": argv, "cwd": str(cwd)})
    start = time.monotonic()
    with (logs / "stdout.log").open("w") as out, (logs / "stderr.log").open("w") as err:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=err,
            text=True,
            start_new_session=True,
        )
        try:
            if prompt is not None:
                process.stdin.write(prompt)
            process.stdin.close()
            while process.poll() is None:
                if stop_file.exists():
                    raise OperatorStop("stop requested")
                if time.monotonic() - start > timeout:
                    raise ProcessTimeout("subprocess timeout")
                time.sleep(0.2)
            if process.returncode:
                raise RuntimeError(f"subprocess exit {process.returncode}; see {logs}")
        finally:
            # Kill the whole group, including children of a timed-out process.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return time.monotonic() - start
