"""Run tab data layer tests (no terminal required)."""

from __future__ import annotations

from pathlib import Path

from rl_adaptive_dbs.tui.run_data import (
    CATEGORY_CLI,
    CATEGORY_DIAGNOSTICS,
    CATEGORY_FIGURE_REPLICATION,
    CATEGORY_TRAINING,
    cycle_folder_id,
    discover_run_recipes,
    filter_folders,
    filter_recipes,
    folder_row_id,
    is_folder_row_id,
    list_run_folders,
    recipes_in_category,
    select_folder,
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


def test_discover_uses_figure_replication_category() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    fig = next(item for item in recipes if item.recipe_id.endswith("2a/plot.py"))
    assert fig.category == CATEGORY_FIGURE_REPLICATION
    assert fig.label.startswith("fig 2a")


def test_list_run_folders_groups_by_category() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    folders = list_run_folders(recipes)
    categories = {item.category for item in folders}
    assert CATEGORY_CLI in categories
    assert CATEGORY_FIGURE_REPLICATION in categories
    assert all(item.count > 0 for item in folders)


def test_recipes_in_category_scopes_cli() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    cli = recipes_in_category(recipes, CATEGORY_CLI)
    assert cli
    assert all(item.category == CATEGORY_CLI for item in cli)


def test_folder_row_ids_round_trip() -> None:
    row_id = folder_row_id(CATEGORY_TRAINING)
    assert is_folder_row_id(row_id)
    assert row_id.startswith("folder:")


def test_filter_folders_by_display_name() -> None:
    folders = list_run_folders(discover_run_recipes(REPO_ROOT))
    filtered = filter_folders(folders, "cli")
    assert filtered
    assert all(item.category == CATEGORY_CLI for item in filtered)


def test_discover_sorts_ship_surface_before_training_and_diagnostics() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    folders = list_run_folders(recipes)
    categories = [item.category for item in folders]
    if CATEGORY_TRAINING in categories and CATEGORY_DIAGNOSTICS in categories:
        assert categories.index(CATEGORY_TRAINING) < categories.index(CATEGORY_DIAGNOSTICS)
    if CATEGORY_FIGURE_REPLICATION in categories and CATEGORY_TRAINING in categories:
        assert categories.index(CATEGORY_FIGURE_REPLICATION) < categories.index(CATEGORY_TRAINING)


def test_filter_recipes_by_category_display_name() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    filtered = filter_recipes(recipes, "figure replication")
    assert filtered
    assert all(item.category == CATEGORY_FIGURE_REPLICATION for item in filtered)


def test_filter_recipes_by_label() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    filtered = filter_recipes(recipes, "info")
    assert filtered
    assert all("info" in item.label.lower() or "info" in item.recipe_id for item in filtered)


def test_select_recipe_fallback() -> None:
    recipes = discover_run_recipes(REPO_ROOT)
    assert select_recipe(recipes, "missing-id") == recipes[0]


def test_select_folder_fallback() -> None:
    folders = list_run_folders(discover_run_recipes(REPO_ROOT))
    assert select_folder(folders, "folder:missing") == folders[0]


def test_cycle_folder_id_wraps() -> None:
    folders = list_run_folders(discover_run_recipes(REPO_ROOT))
    if len(folders) < 2:
        return
    first = folder_row_id(folders[0].category)
    second = cycle_folder_id(folders, first, 1)
    assert second == folder_row_id(folders[1].category)
