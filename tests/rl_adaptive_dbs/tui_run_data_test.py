"""Run tab data layer tests (no terminal required)."""

from __future__ import annotations

from pathlib import Path

from rl_adaptive_dbs.tui.run_data import (
    discover_run_recipes,
    filter_recipes,
    select_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_discover_includes_cli_recipes() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    ids = {item.recipe_id for item in recipes}
    assert "cli/train-ddpg-paper" in ids
    assert "cli/info" in ids


def test_discover_includes_repo_scripts() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    ids = {item.recipe_id for item in recipes}
    assert any(item.startswith("scripts/") for item in ids)


def test_filter_recipes_by_label() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    filtered = filter_recipes(recipes, "info")
    assert filtered
    assert all("info" in item.label.lower() or "info" in item.recipe_id for item in filtered)


def test_select_recipe_fallback() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    assert select_recipe(recipes, "missing-id") == recipes[0]
