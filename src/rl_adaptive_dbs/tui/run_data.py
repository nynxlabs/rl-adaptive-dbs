"""Run tab recipe catalog and script discovery."""

from __future__ import annotations

import ast
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rl_adaptive_dbs.paths import find_repo_root
from rl_adaptive_dbs.tui.run_launch import command_text

# Stable ids for sorting/filtering; display names are user-facing in the table.
CATEGORY_FIGURE_REPLICATION = "figure-replication"
CATEGORY_CLI = "cli"
CATEGORY_TRAINING = "training"
CATEGORY_DIAGNOSTICS = "diagnostics"
CATEGORY_REPLICATION = "replication"

CATEGORY_ORDER: tuple[str, ...] = (
    CATEGORY_FIGURE_REPLICATION,
    CATEGORY_CLI,
    CATEGORY_TRAINING,
    CATEGORY_DIAGNOSTICS,
    CATEGORY_REPLICATION,
)

CATEGORY_DISPLAY: dict[str, str] = {
    CATEGORY_FIGURE_REPLICATION: "figure replication",
    CATEGORY_CLI: "CLI",
    CATEGORY_TRAINING: "training",
    CATEGORY_DIAGNOSTICS: "diagnostics",
    CATEGORY_REPLICATION: "replication",
}

FOLDER_ROW_PREFIX = "folder:"


def category_display_name(category: str) -> str:
    return CATEGORY_DISPLAY.get(category, category)


def folder_row_id(category: str) -> str:
    return f"{FOLDER_ROW_PREFIX}{category}"


def is_folder_row_id(row_id: str) -> bool:
    return row_id.startswith(FOLDER_ROW_PREFIX)


def folder_id_from_row(row_id: str) -> str:
    return row_id.removeprefix(FOLDER_ROW_PREFIX)


@dataclass(frozen=True)
class RunFolder:
    """A browsable category folder on the Run tab root."""

    category: str
    label: str
    count: int


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
            category=CATEGORY_CLI,
            label="train ddpg paper (seed 0)",
            argv=tuple([*rl_dbs, "train", "--controller", "ddpg", "--variant", "paper", "--seeds", "0"]),
            description="Mehregan DDPG paper replication, single seed.",
        ),
        RunRecipe(
            recipe_id="cli/benchmark-mehregan-smoke",
            category=CATEGORY_CLI,
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
            category=CATEGORY_CLI,
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


def _figure_recipe_label(path: Path) -> str:
    doc = _script_docstring(path)
    match = re.search(r"Figure\s+(\S+)\s*[—–-]\s*(.+)", doc)
    if match:
        panel = match.group(1).rstrip(".")
        title = match.group(2).strip().rstrip(".")
        return f"fig {panel} — {title}"
    parts = path.parts
    if "papers" in parts:
        idx = parts.index("papers")
        if idx + 2 < len(parts):
            return f"fig {parts[idx + 2]}"
    return path.parent.name


def _script_argv(repo_root: Path, rel: Path, *, plant_heavy: bool = False) -> list[str]:
    prefix = _uv_prefix()
    rel_posix = rel.as_posix()
    if plant_heavy and shutil.which("uv"):
        return [
            *prefix,
            "python",
            "-m",
            "rl_adaptive_dbs.run",
            rel_posix,
        ]
    if prefix[0] == "uv":
        return [*prefix, "python", rel_posix]
    return [*prefix, rel_posix]


def _discover_scripts(
    repo_root: Path,
    pattern: str,
    *,
    category: str,
    plant_heavy: bool = False,
    label_fn: Callable[[Path], str] | None = None,
) -> list[RunRecipe]:
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
        recipe_id = rel.as_posix()
        label = label_fn(path) if label_fn is not None else path.stem.replace("_", " ")
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


def _sort_recipes(recipes: list[RunRecipe]) -> list[RunRecipe]:
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}

    def sort_key(item: RunRecipe) -> tuple[int, str, str]:
        return (order.get(item.category, len(CATEGORY_ORDER)), item.label.lower(), item.recipe_id)

    return sorted(recipes, key=sort_key)


def discover_run_recipes(repo_root: Path | None = None) -> list[RunRecipe]:
    """Built-in CLI recipes plus discoverable repo scripts."""
    root = (repo_root or find_repo_root()).resolve()
    recipes: list[RunRecipe] = []
    recipes.extend(_cli_recipes())
    if not (root / "scripts").is_dir():
        return _sort_recipes(recipes)

    recipes.extend(
        _discover_scripts(
            root,
            "figures/**/plot.py",
            category=CATEGORY_FIGURE_REPLICATION,
            label_fn=_figure_recipe_label,
        )
    )
    recipes.extend(
        _discover_scripts(
            root,
            "training/run_*.py",
            category=CATEGORY_TRAINING,
            plant_heavy=True,
        )
    )
    recipes.extend(
        _discover_scripts(
            root,
            "probes/*.py",
            category=CATEGORY_DIAGNOSTICS,
            plant_heavy=True,
        )
    )
    recipes.extend(
        _discover_scripts(root, "replication/replicate_*.py", category=CATEGORY_REPLICATION)
    )
    recipes.extend(
        _discover_scripts(root, "replication/check_*.py", category=CATEGORY_REPLICATION)
    )

    seen: set[str] = set()
    unique: list[RunRecipe] = []
    for item in recipes:
        if item.recipe_id in seen:
            continue
        seen.add(item.recipe_id)
        unique.append(item)
    return _sort_recipes(unique)


def list_run_folders(recipes: list[RunRecipe]) -> list[RunFolder]:
    """Category folders shown at the Run tab root."""
    counts: dict[str, int] = {}
    for item in recipes:
        counts[item.category] = counts.get(item.category, 0) + 1

    folders: list[RunFolder] = []
    for category in CATEGORY_ORDER:
        count = counts.get(category, 0)
        if count:
            folders.append(
                RunFolder(
                    category=category,
                    label=category_display_name(category),
                    count=count,
                )
            )
    for category in sorted(counts):
        if category in CATEGORY_ORDER:
            continue
        folders.append(
            RunFolder(
                category=category,
                label=category_display_name(category),
                count=counts[category],
            )
        )
    return folders


def recipes_in_category(recipes: list[RunRecipe], category: str) -> list[RunRecipe]:
    return [item for item in recipes if item.category == category]


def filter_folders(folders: list[RunFolder], needle: str) -> list[RunFolder]:
    if not needle.strip():
        return list(folders)
    key = needle.strip().lower()
    return [
        item
        for item in folders
        if key in item.label.lower()
        or key in item.category.lower()
        or key in category_display_name(item.category).lower()
    ]


def filter_recipes(recipes: list[RunRecipe], needle: str) -> list[RunRecipe]:
    if not needle.strip():
        return list(recipes)
    key = needle.strip().lower()
    return [
        item
        for item in recipes
        if key in item.label.lower()
        or key in item.category.lower()
        or key in category_display_name(item.category).lower()
        or key in item.recipe_id.lower()
        or key in item.command_preview.lower()
    ]


def _truncate_command(preview: str, *, limit: int = 56) -> str:
    if len(preview) <= limit:
        return preview
    return preview[:26] + "…" + preview[-26:]


def folder_table_rows(folders: list[RunFolder]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in folders:
        suffix = "recipe" if item.count == 1 else "recipes"
        rows.append((item.label, f"{item.count} {suffix}"))
    return rows


def recipe_table_rows(recipes: list[RunRecipe]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in recipes:
        rows.append((item.label, _truncate_command(item.command_preview)))
    return rows


def select_recipe(recipes: list[RunRecipe], recipe_id: str | None) -> RunRecipe | None:
    if not recipes or recipe_id is None:
        return recipes[0] if recipes else None
    for item in recipes:
        if item.recipe_id == recipe_id:
            return item
    return recipes[0]


def select_folder(folders: list[RunFolder], row_id: str | None) -> RunFolder | None:
    if not folders or row_id is None:
        return folders[0] if folders else None
    category = folder_id_from_row(row_id) if is_folder_row_id(row_id) else row_id
    for item in folders:
        if item.category == category:
            return item
    return folders[0]


def cycle_recipe_id(recipes: list[RunRecipe], active_id: str | None, delta: int) -> str | None:
    if not recipes:
        return None
    ids = [item.recipe_id for item in recipes]
    if active_id not in ids:
        return ids[0]
    index = (ids.index(active_id) + delta) % len(ids)
    return ids[index]


def cycle_folder_id(folders: list[RunFolder], active_id: str | None, delta: int) -> str | None:
    if not folders:
        return None
    ids = [folder_row_id(item.category) for item in folders]
    if active_id not in ids:
        return ids[0]
    index = (ids.index(active_id) + delta) % len(ids)
    return ids[index]


def run_status_line(
    *,
    browse_category: str | None,
    folder_count: int,
    visible_count: int,
    selected_label: str | None = None,
    last_launch: str | None = None,
) -> str:
    if browse_category is None:
        parts = [f"{folder_count} folders"]
    else:
        parts = [
            f"{category_display_name(browse_category)}/",
            f"{visible_count} recipes",
        ]
    if selected_label:
        parts.append(f"selected: {selected_label}")
    if last_launch:
        parts.append(last_launch)
    return "  |  ".join(parts)


def run_empty_message(repo_root: Path) -> str:
    return (
        f"No runnable recipes under {repo_root}/scripts/.\n"
        "Built-in rl-dbs CLI recipes still appear when the repo root is found."
    )


def run_root_hints() -> str:
    return "Enter: open folder  |  /: filter folders  |  Esc: back / clear filter"


def run_folder_hints() -> str:
    return (
        "Enter or x: launch detached  |  Esc: back to folders"
        "  |  f: follow mode in confirm dialog  |  output → artifacts/tui-runs/"
    )
