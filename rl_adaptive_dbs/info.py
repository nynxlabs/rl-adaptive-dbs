"""Repository introspection for ``rl-dbs info``."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from benchmarks.git import git_commit_short
from benchmarks.suite import default_suites_dir, find_repo_root
from rl_adaptive_dbs.user_config import resolve_config

CONTROLLER_VARIANTS: dict[str, tuple[str, ...]] = {
    "ddpg": ("paper", "init-30hz", "ptq-fp16", "ptq-int8", "qat"),
    "snn": ("paper",),
    "sea_dbs": ("paper", "baseline", "baseline-pm", "baseline-gs"),
    "baseline": ("none", "cdbs-130hz", "periodic-45hz", "periodic-30hz"),
}


def list_suite_names(repo_root: Path | None = None) -> list[str]:
    root = repo_root or find_repo_root()
    suites_dir = default_suites_dir(root)
    if not suites_dir.is_dir():
        return []
    return sorted(path.stem for path in suites_dir.glob("*.yaml"))


def env_info() -> dict[str, Any]:
    resolved = resolve_config()
    cfg = resolved.env
    return {
        "step_duration_s": cfg.step_duration_s,
        "max_episode_steps": cfg.max_episode_steps,
        "beta_threshold": cfg.beta_threshold,
        "reward_scale": cfg.reward_scale,
        "observation_scale": cfg.observation_scale,
        "state_length": cfg.state_length,
        "biomarker_band_hz": list(resolved.biomarker_band_hz),
    }


def plant_info() -> dict[str, Any]:
    resolved = resolve_config()
    cfg = resolved.plant
    return {
        "backend": resolved.plant_backend,
        "dt_ms": cfg.dt_ms,
        "pd": cfg.pd,
        "corstim": cfg.corstim,
        "neurons_per_region": cfg.neurons_per_region,
    }


def version_info(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    return {
        "package": "rl-adaptive-dbs",
        "version": version("rl-adaptive-dbs"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit_short(root),
    }


def controllers_info() -> dict[str, Any]:
    return {
        name: {"variants": list(variants)}
        for name, variants in CONTROLLER_VARIANTS.items()
    }


def build_info_payload(
    topic: str | None = None,
    *,
    controller: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or find_repo_root()
    if topic in (None, ""):
        return {
            "controllers": controllers_info(),
            "suites": list_suite_names(root),
            "env": env_info(),
            "plant": plant_info(),
            "version": version_info(root),
        }
    if topic == "controllers":
        return {"controllers": controllers_info()}
    if topic == "variants":
        if controller is None:
            return {"variants": CONTROLLER_VARIANTS}
        if controller not in CONTROLLER_VARIANTS:
            msg = f"unknown controller {controller!r}"
            raise KeyError(msg)
        return {"controller": controller, "variants": list(CONTROLLER_VARIANTS[controller])}
    if topic == "suites":
        return {"suites": list_suite_names(root)}
    if topic == "env":
        return {"env": env_info()}
    if topic == "plant":
        return {"plant": plant_info()}
    if topic == "version":
        return {"version": version_info(root)}
    msg = f"unknown info topic {topic!r}"
    raise KeyError(msg)


def format_info_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    if "controllers" in payload and "suites" in payload:
        lines.append("Controllers:")
        for name, meta in payload["controllers"].items():
            variants = ", ".join(meta["variants"])
            lines.append(f"  {name}: {variants}")
        lines.append("")
        lines.append("Suites:")
        for suite in payload["suites"]:
            lines.append(f"  {suite}")
        lines.append("")
        lines.append("Env (Mehregan):")
        for key, value in payload["env"].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        lines.append("Plant:")
        for key, value in payload["plant"].items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        ver = payload["version"]
        lines.append(
            f"Version: {ver['version']}  python={ver['python']}  git={ver.get('git_commit') or '?'}"
        )
        return "\n".join(lines)

    if "controllers" in payload and len(payload) == 1:
        lines.append("Controllers:")
        for name, meta in payload["controllers"].items():
            lines.append(f"  {name}: {', '.join(meta['variants'])}")
        return "\n".join(lines)
    if "suites" in payload:
        return "Suites:\n" + "\n".join(f"  {s}" for s in payload["suites"])
    if "env" in payload:
        return "Env:\n" + "\n".join(f"  {k}: {v}" for k, v in payload["env"].items())
    if "plant" in payload:
        return "Plant:\n" + "\n".join(f"  {k}: {v}" for k, v in payload["plant"].items())
    if "version" in payload:
        ver = payload["version"]
        return (
            f"rl-adaptive-dbs {ver['version']}\n"
            f"python {ver['python']}\n"
            f"platform {ver['platform']}\n"
            f"git {ver.get('git_commit') or '?'}"
        )
    if "variants" in payload and "controller" in payload:
        return f"{payload['controller']}: " + ", ".join(payload["variants"])
    return str(payload)
