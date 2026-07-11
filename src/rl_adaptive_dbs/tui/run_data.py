"""Run tab recipe catalog and script discovery."""

from __future__ import annotations

import ast
import shutil
from dataclasses import dataclass
from pathlib import Path

from rl_adaptive_dbs.paths import find_repo_root
from rl_adaptive_dbs.tui.run_launch import command_text


@dataclass(frozen=True)
class RunRecipe:
    """A launchable command the TUI can start detached."""

    recipe_id: str
    category: str
    label: str
    argv: tuple[str, ...]
    description: str = ""
    log_stem: str | None = None

    @property
    def command_preview(self) -> str:
        return command_text(list(self.argv))

    def log_recipe_id(self) -> str:
        return self.log_stem or self.recipe_id


def _uv_prefix() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "run"]
    return [shutil.which("python") or "python"]


def _cli_recipes() -> list[RunRecipe]:
    prefix = _uv_prefix()
    rl_dbs = [*prefix, "rl-dbs"]
    return [
        RunRecipe(
            recipe_id="cli/train-ddpg-paper",
            category="cli",
            label="train ddpg paper (seed 0)",
            argv=tuple([*rl_dbs, "train", "--controller", "ddpg", "--variant", "paper", "--seeds", "0"]),
            description="Mehregan DDPG paper replication, single seed.",
        ),
        RunRecipe(
            recipe_id="cli/benchmark-mehregan-smoke",
            category="cli",
            label="benchmark mehregan_eval_smoke",
            argv=tuple(
                [
                    *rl_dbs,
                    "benchmark",
                    "--suite",
                    "mehregan_eval_smoke",
                    "--results-dir",
                    "results/",
                ]
            ),
            description="Smoke benchmark suite (fast sanity check).",
        ),
        RunRecipe(
            recipe_id="cli/info",
            category="cli",
            label="info controllers",
            argv=tuple([*rl_dbs, "info", "controllers"]),
            description="List registered controllers (quick, <30s).",
        ),
    ]


def _script_docstring(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return ""
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    return first[:120]


def _looks_like_entry_point(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text:
        return True
    if "run_main(" in text or "argparse.ArgumentParser" in text:
        return True
    return False


def _script_argv(repo_root: Path, rel: Path, *, plant_heavy: bool = False) -> list[str]:
    prefix = _uv_prefix()
    rel_posix = rel.as_posix()
    if plant_heavy and shutil.which("uv"):
        return [
            *prefix,
            "python",
            "-m",
            "rl_adaptive_dbs.run",
            "--max-threads",
            "3",
            rel_posix,
        ]
    if prefix[0] == "uv":
        return [*prefix, "python", rel_posix]
    return [*prefix, rel_posix]


def _discover_scripts(repo_root: Path, pattern: str, *, category: str, plant_heavy: bool = False) -> list[RunRecipe]:
    scripts_dir = repo_root / "scripts"
    recipes: list[RunRecipe] = []
    for path in sorted(scripts_dir.glob(pattern)):
        if not path.is_file() or path.name.startswith("_"):
            continue
        if not _looks_like_entry_point(path):
            continue
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        recipe_id = f"scripts/{rel.as_posix()}"
        label = path.stem.replace("_", " ")
        recipes.append(
            RunRecipe(
                recipe_id=recipe_id,
                category=category,
                label=label,
                argv=tuple(_script_argv(repo_root, rel, plant_heavy=plant_heavy)),
                description=_script_docstring(path),
            )
        )
    return recipes


def discover_run_recipes(repo_root: Path | None = None) -> list[RunRecipe]:
    """Built-in CLI recipes plus discoverable repo scripts."""
    root = (repo_root or find_repo_root()).resolve()
    recipes: list[RunRecipe] = []
    recipes.extend(_cli_recipes())
    if not (root / "scripts").is_dir():
        return recipes

    recipes.extend(_discover_scripts(root, "training/run_*.py", category="training"))
    recipes.extend(_discover_scripts(root, "replication/replicate_*.py", category="replication"))
    recipes.extend(_discover_scripts(root, "replication/check_*.py", category="replication"))
    recipes.extend(_discover_scripts(root, "figures/**/plot.py", category="figures"))
    recipes.extend(
        _discover_scripts(root, "probes/run_*.py", category="probes", plant_heavy=True)
    )

    seen: set[str] = set()
    unique: list[RunRecipe] = []
    for item in recipes:
        if item.recipe_id in seen:
            continue
        seen.add(item.recipe_id)
        unique.append(item)
    return unique


def filter_recipes(recipes: list[RunRecipe], needle: str) -> list[RunRecipe]:
    if not needle.strip():
        return list(recipes)
    key = needle.strip().lower()
    return [
        item
        for item in recipes
        if key in item.label.lower()
        or key in item.category.lower()
        or key in item.recipe_id.lower()
        or key in item.command_preview.lower()
    ]


def recipe_table_rows(recipes: list[RunRecipe]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for item in recipes:
        preview = item.command_preview
        if len(preview) > 56:
            preview = preview[:26] + "…" + preview[-26:]
        rows.append((item.category, item.label, preview))
    return rows


def select_recipe(recipes: list[RunRecipe], recipe_id: str | None) -> RunRecipe | None:
    if not recipes or recipe_id is None:
        return recipes[0] if recipes else None
    for item in recipes:
        if item.recipe_id == recipe_id:
            return item
    return recipes[0]


def cycle_recipe_id(recipes: list[RunRecipe], active_id: str | None, delta: int) -> str | None:
    if not recipes:
        return None
    ids = [item.recipe_id for item in recipes]
    if active_id not in ids:
        return ids[0]
    index = (ids.index(active_id) + delta) % len(ids)
    return ids[index]


def run_status_line(
    recipes: list[RunRecipe],
    active: RunRecipe | None,
    *,
    last_launch: str | None = None,
) -> str:
    if not recipes:
        return "No launch recipes (scripts/ missing?)"
    if active is None:
        return f"{len(recipes)} recipes"
    parts = [f"{len(recipes)} recipes", f"selected: {active.category}/{active.label}"]
    if last_launch:
        parts.append(last_launch)
    return "  |  ".join(parts)


def run_empty_message(repo_root: Path) -> str:
    return (
        f"No runnable recipes under {repo_root}/scripts/.\n"
        "Built-in rl-dbs CLI recipes still appear when the repo root is found."
    )
