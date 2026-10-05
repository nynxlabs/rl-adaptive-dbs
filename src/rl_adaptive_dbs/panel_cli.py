"""``rl-dbs panel`` — launch, compare, and promote paper-figure panel runs.

``run``      Start ``scripts/figures/papers/<paper>/<panel>/plot.py`` detached: its own systemd
             scope (escapes agent-service cgroups), a private ``tmux -L`` server, ``nice``,
             one math thread, a run log with header/exit markers, and a cgroup check.
             ``--candidate NAME`` sends every output to ``candidates/NAME/`` and skips promote.
``compare``  Table of candidates (and the promoted run) against the panel's gates.
``promote``  Install a finished candidate's series/checkpoint as the panel's own outputs and
             replot through the normal promote path, recording where they came from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rl_adaptive_dbs import panel as _panel
from rl_adaptive_dbs.run_log_meta import (
    RunLogMeta,
    utc_now_iso,
    write_run_log_exit,
    write_run_log_header,
)

REPO_ROOT = _panel.REPO_ROOT
PAPERS_DIR = REPO_ROOT / "scripts" / "figures" / "papers"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "figures" / "papers"
LOGS_DIR = REPO_ROOT / "logs"

# Output flags redirected into the candidate directory when the panel script supports them.
_CANDIDATE_OUTPUTS = {
    "--series": "series.json",
    "--checkpoint": "checkpoint.pt",
    "--manifest": "manifest.json",
    "--eval-json": "eval.json",
    "--qat-checkpoint": "qat.pt",
    "--out": "replication_v1.png",
}
_NO_PROMOTE_FLAGS = ("--no-update-docs", "--no-promote")
_PROMOTE_FLAGS = ("--export-notes", "--update-report")
_FORBIDDEN_CGROUPS = ("synara.service", "hermes", "cursor")
_INSTALLABLE = ("series.json", "checkpoint.pt", "checkpoint.metrics.json")


def _panel_script(panel_id: str) -> Path:
    script = PAPERS_DIR / panel_id / "plot.py"
    if not script.is_file():
        msg = f"no panel script at {script.relative_to(REPO_ROOT)} (expected <paper>/<panel>)"
        raise SystemExit(msg)
    return script


def _panel_flags(script: Path) -> set[str]:
    """Long options the panel script's argparse accepts (from ``--help``)."""
    out = subprocess.run(
        [sys.executable, "-m", "rl_adaptive_dbs.run", str(script), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    return set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", out.stdout))


def _has_flag(args: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def _session_name(panel_id: str, candidate: str | None) -> str:
    raw = f"{panel_id.replace('/', '-')}-{candidate or 'main'}"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", raw)


def candidate_dir(panel_id: str, candidate: str) -> Path:
    return ARTIFACTS_DIR / panel_id / "candidates" / candidate


def build_panel_args(
    panel_id: str,
    script_args: list[str],
    *,
    candidate: str | None,
    flags: set[str],
    abort_on_fail: bool = True,
) -> list[str]:
    """Panel argv with candidate output redirection, promote flags, and safe defaults."""
    args = list(script_args)
    if candidate:
        out_dir = candidate_dir(panel_id, candidate)
        for flag, name in _CANDIDATE_OUTPUTS.items():
            if flag in flags and not _has_flag(args, flag):
                args += [flag, str(out_dir / name)]
        for flag in _NO_PROMOTE_FLAGS:
            if flag in flags and not _has_flag(args, flag):
                args.append(flag)
        if abort_on_fail and "--abort-on-fail" in flags and not _has_flag(args, "--abort-on-fail"):
            args.append("--abort-on-fail")
    else:
        for flag in _PROMOTE_FLAGS:
            if flag in flags and not _has_flag(args, flag):
                args.append(flag)
    if "--parallel-series" in flags and not _has_flag(args, "--parallel-series"):
        args += ["--parallel-series", "1"]
    return args


def _exec_command(log: Path, session: str, pid_file: Path, script: Path, panel_args: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "rl_adaptive_dbs.panel_cli",
        "_exec",
        "--log",
        str(log),
        "--session",
        session,
        "--pid-file",
        str(pid_file),
        "--",
        sys.executable,
        "-m",
        "rl_adaptive_dbs.run",
        "--max-threads",
        "1",
        str(script),
        *panel_args,
    ]


def _slice_args() -> list[str]:
    """Run training in ``heavy.slice`` when the host defines one (shared CPU/RAM caps)."""
    slice_name = os.environ.get("RL_DBS_SLICE", "heavy.slice")
    if not slice_name:
        return []
    probe = subprocess.run(
        ["systemctl", "--user", "cat", slice_name],
        capture_output=True,
        check=False,
    )
    return [f"--slice={slice_name}"] if probe.returncode == 0 else []


def _launch_argv(session: str, inner: list[str]) -> list[str]:
    """systemd scope → private tmux server → setsid/nohup/nice; degrade when tools are missing."""
    shell = f"cd {shlex.quote(str(REPO_ROOT))} && setsid nohup nice -n 10 " + shlex.join(inner) + " < /dev/null"
    if shutil.which("tmux"):
        argv = ["tmux", "-L", session, "new-session", "-d", "-s", session, shell]
        if shutil.which("systemd-run") and sys.platform.startswith("linux"):
            argv = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={session}", *_slice_args(), "--", *argv]
        return argv
    return ["sh", "-c", shell + " &"]


def _cgroup_of(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        return None


def watch_prompt(*, panel_id: str, session: str, log: Path, manifest: Path) -> str:
    rel = lambda p: p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else str(p)  # noqa: E731
    return (
        f"Watch tmux socket {session} (`tmux -L {session}`) for panel {panel_id}.\n"
        f"Log: {rel(log)} (main checkout). Exit marker line starts with '# rl-dbs-run-exit:' "
        "(exit 0 pass, 1 gate fail, 2 missing input/refused, 3 early abort).\n"
        f"Manifest: {rel(manifest)} — gates_pass is the success criterion; "
        f"{rel(manifest.with_name(manifest.stem + '.partial.json'))} has checkpoint probes.\n"
        "Follow AGENTS.md § If you are a train/eval watch automation. No foreground sleep between polls."
    )


def cmd_run(ns: argparse.Namespace) -> int:
    script = _panel_script(ns.panel)
    flags = _panel_flags(script)
    panel_args = build_panel_args(
        ns.panel,
        ns.script_args,
        candidate=ns.candidate,
        flags=flags,
        abort_on_fail=not ns.no_abort,
    )
    session = _session_name(ns.panel, ns.candidate)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = LOGS_DIR / f"{session}.log"
    pid_file = LOGS_DIR / f"{session}.pid"
    inner = _exec_command(log, session, pid_file, script, panel_args)
    if ns.candidate:
        candidate_dir(ns.panel, ns.candidate).mkdir(parents=True, exist_ok=True)
        manifest = candidate_dir(ns.panel, ns.candidate) / "manifest.json"
    else:
        manifest = ARTIFACTS_DIR / ns.panel / "manifest.json"

    if ns.foreground:
        print("$ " + shlex.join(inner[inner.index("--") + 1 :]), flush=True)
        return subprocess.call(inner, cwd=REPO_ROOT)

    argv = _launch_argv(session, inner)
    if ns.dry_run:
        print(shlex.join(argv))
        return 0
    if shutil.which("tmux") and subprocess.run(
        ["tmux", "-L", session, "has-session", "-t", session], capture_output=True, check=False
    ).returncode == 0:
        print(f"tmux session {session} already running (tmux -L {session} attach -t {session})", file=sys.stderr)
        return _panel.EXIT_MISSING_INPUT
    pid_file.unlink(missing_ok=True)
    subprocess.run(argv, cwd=REPO_ROOT, check=True)

    pid = None
    for _ in range(60):
        if pid_file.is_file():
            pid = int(pid_file.read_text(encoding="utf-8").strip() or 0) or None
            if pid:
                break
        time.sleep(0.25)
    print(f"session: {session}  (tmux -L {session} attach -t {session})")
    print(f"log:     {log}")
    print(f"manifest:{manifest}")
    if pid is None:
        print("WARNING: run process did not report a pid yet; check the log", file=sys.stderr)
    else:
        cgroup = _cgroup_of(pid)
        bad = [name for name in _FORBIDDEN_CGROUPS if cgroup and name in cgroup]
        print(f"pid:     {pid}  cgroup: {cgroup}")
        if bad:
            print(f"ERROR: run landed inside {bad}; it will die with that service. Kill it and relaunch.", file=sys.stderr)
            return _panel.EXIT_MISSING_INPUT
    print("\n--- watch automation prompt ---")
    print(watch_prompt(panel_id=ns.panel, session=session, log=log, manifest=manifest))
    return 0


def cmd_exec(ns: argparse.Namespace) -> int:
    """Inner runner: log header, child process (pid file), exit marker."""
    log = Path(ns.log)
    command = ns.command[1:] if ns.command and ns.command[0] == "--" else ns.command
    write_run_log_header(
        log,
        RunLogMeta(
            pid=os.getpid(),
            command=shlex.join(command),
            started_at=utc_now_iso(),
            tmux_session=ns.session,
            pid_file=ns.pid_file,
            log_file=str(log),
            repo_root=str(REPO_ROOT),
        ),
    )
    env = {**os.environ, "RL_DBS_MAX_THREADS": os.environ.get("RL_DBS_MAX_THREADS", "1")}
    with log.open("a", encoding="utf-8") as handle:
        child = subprocess.Popen(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
        if ns.pid_file:
            Path(ns.pid_file).write_text(f"{child.pid}\n", encoding="utf-8")
        code = child.wait()
    write_run_log_exit(log, exit_code=code)
    return code


def _manifest_rows(panel_id: str) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    main = ARTIFACTS_DIR / panel_id / "manifest.json"
    if main.is_file():
        rows.append(("(promoted)", main))
    cand_root = ARTIFACTS_DIR / panel_id / "candidates"
    if cand_root.is_dir():
        for path in sorted(cand_root.glob("*/manifest.json")):
            rows.append((path.parent.name, path))
    return rows


def _failed_keys(manifest: dict[str, Any]) -> list[str]:
    failed: list[str] = list(manifest.get("shape_failed") or [])
    gates = manifest.get("gates") or {}
    blocks = [gates] + [v for v in gates.values() if isinstance(v, dict)]
    for block in blocks:
        for key, value in block.items():
            if value is False and key not in {"pass", "shape_pass", "all_pass", "smoke_override"} and key not in failed:
                failed.append(key)
    return failed


def config_diff(base: dict[str, Any] | None, other: dict[str, Any] | None) -> dict[str, Any]:
    if not base or not other:
        return {}
    return {k: other[k] for k in sorted(other) if base.get(k) != other.get(k)}


def cmd_compare(ns: argparse.Namespace) -> int:
    rows = _manifest_rows(ns.panel)
    if not rows:
        print(f"no manifests under {ARTIFACTS_DIR / ns.panel}", file=sys.stderr)
        return _panel.EXIT_MISSING_INPUT
    loaded = [(name, json.loads(path.read_text(encoding="utf-8"))) for name, path in rows]
    base_cfg = next((m.get("config") for name, m in loaded if name == "(promoted)" and m.get("config")), None)
    print(f"{'candidate':<22} {'status':<10} {'pass':<6} {'#fail':>5}  failed / config diff")
    for name, manifest in loaded:
        status = f"abort@{manifest['aborted_at_episode']}" if manifest.get("aborted") else "done"
        verdict = _panel.gates_pass(manifest)
        failed = _failed_keys(manifest)
        shown = ", ".join(failed[:4]) + (" …" if len(failed) > 4 else "")
        print(f"{name:<22} {status:<10} {verdict!s:<6} {len(failed):>5}  {shown}")
        diff = config_diff(base_cfg, manifest.get("config"))
        if diff and name != "(promoted)":
            print(f"{'':<22} {'':<10} {'':<6} {'':>5}  config: {json.dumps(diff)}")
    return 0


def cmd_promote(ns: argparse.Namespace) -> int:
    src = candidate_dir(ns.panel, ns.candidate)
    manifest_path = src / "manifest.json"
    if not manifest_path.is_file():
        print(f"no candidate manifest at {manifest_path}", file=sys.stderr)
        return _panel.EXIT_MISSING_INPUT
    cand_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cand_manifest.get("aborted") or cand_manifest.get("smoke"):
        print("refusing to promote an aborted or smoke candidate", file=sys.stderr)
        return _panel.EXIT_MISSING_INPUT
    present = [name for name in _INSTALLABLE if (src / name).is_file()]
    if "series.json" not in present:
        print(
            "candidate has no series.json; this panel promotes by re-running its script with the "
            f"winning flags: {shlex.join((cand_manifest.get('provenance') or {}).get('argv') or [])}",
            file=sys.stderr,
        )
        return _panel.EXIT_MISSING_INPUT
    dest = ARTIFACTS_DIR / ns.panel
    for name in present:
        shutil.copy2(src / name, dest / name)
        print(f"installed {name} from candidate {ns.candidate}")
    script = _panel_script(ns.panel)
    flags = _panel_flags(script)
    args = ["--plot-only", *build_panel_args(ns.panel, [], candidate=None, flags=flags)]
    code = subprocess.call(
        [sys.executable, "-m", "rl_adaptive_dbs.run", "--max-threads", "1", str(script), *args],
        cwd=REPO_ROOT,
    )
    promoted = dest / "manifest.json"
    if promoted.is_file():
        data = json.loads(promoted.read_text(encoding="utf-8"))
        data["promoted_from"] = {
            "candidate": ns.candidate,
            "manifest": str(manifest_path),
            "provenance": cand_manifest.get("provenance"),
        }
        promoted.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return code


def build_parser(sub: argparse._SubParsersAction | None = None) -> argparse.ArgumentParser:
    if sub is None:
        parser = argparse.ArgumentParser(prog="rl-dbs panel", description=__doc__.split("\n\n")[0])
    else:
        parser = sub.add_parser("panel", help="Launch, compare, and promote paper-figure panel runs")
    psub = parser.add_subparsers(dest="panel_command", required=True)

    run = psub.add_parser("run", help="Start a panel run detached (scope + tmux + nice + 1 thread)")
    run.add_argument("panel", help="<paper>/<panel>, e.g. nguyen/4")
    run.add_argument("--candidate", help="Write outputs to candidates/NAME/ and skip promote")
    run.add_argument("--no-abort", action="store_true", help="Do not add --abort-on-fail for candidates")
    run.add_argument("--foreground", action="store_true", help="Run in this terminal (no tmux/scope)")
    run.add_argument("--dry-run", action="store_true", help="Print the launch command only")
    run.set_defaults(script_args=[])
    run.epilog = "Pass plot.py arguments after a literal --, e.g. rl-dbs panel run nguyen/4 -- --episodes 500"

    compare = psub.add_parser("compare", help="Candidates vs gates for one panel")
    compare.add_argument("panel")

    promote = psub.add_parser("promote", help="Install a candidate's outputs and replot with promote")
    promote.add_argument("panel")
    promote.add_argument("candidate")

    exec_ = psub.add_parser("_exec")
    exec_.add_argument("--log", required=True)
    exec_.add_argument("--session")
    exec_.add_argument("--pid-file")
    exec_.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def split_script_args(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``panel run … -- <plot.py args>`` so argparse never sees the plot.py arguments."""
    if "run" in argv and "--" in argv and argv.index("--") > argv.index("run"):
        cut = argv.index("--")
        return argv[:cut], argv[cut + 1 :]
    return argv, []


def dispatch(ns: argparse.Namespace) -> int:
    handlers = {"run": cmd_run, "compare": cmd_compare, "promote": cmd_promote, "_exec": cmd_exec}
    return handlers[ns.panel_command](ns)


def main(argv: list[str] | None = None) -> int:
    head, tail = split_script_args(list(sys.argv[1:] if argv is None else argv))
    ns = build_parser().parse_args(head)
    if tail:
        ns.script_args = tail
    return dispatch(ns)


if __name__ == "__main__":
    raise SystemExit(main())
