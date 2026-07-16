"""``rl-dbs-tui`` Textual application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual import events
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.actions import SkipAction
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from benchmarks.loader import filter_runs
from rl_adaptive_dbs.tui.data import (
    refresh_suites,
    select_suite,
    suite_status_line,
)
from rl_adaptive_dbs.tui.eval_data import (
    EvalRun,
    cross_paper_warning,
    cycle_eval_run_id,
    eval_empty_message,
    eval_run_table_rows,
    eval_status_line,
    eval_summary_lines,
    load_eval_context,
    p_beta_sparkline_data,
    select_eval_run,
)
from rl_adaptive_dbs.tui.launch_follow import (
    LAUNCH_FOLLOW_LOGS,
    LAUNCH_FOLLOW_NONE,
    LAUNCH_FOLLOW_TERMINAL,
    cycle_dialog_follow,
    initial_dialog_follow,
    launch_follow_label,
)
from rl_adaptive_dbs.tui.logs_data import (
    LogFile,
    bookmarks_file,
    cycle_log_file,
    discover_log_files,
    filter_log_files,
    log_file_rows,
    log_row_key,
    logs_empty_message,
    logs_hints_line,
    logs_status_line,
    select_log_file,
    tail_lines,
    toggle_bookmark,
)
from rl_adaptive_dbs.paths import find_repo_root
from rl_adaptive_dbs.tui.reload import RESTART_EXIT_CODE
from rl_adaptive_dbs.tui.run_data import (
    RunRecipe,
    cycle_recipe_id,
    discover_run_recipes,
    filter_recipes,
    recipe_table_rows,
    run_empty_message,
    run_status_line,
    select_recipe,
)
from rl_adaptive_dbs.tui.run_launch import (
    LaunchResult,
    launch_detached,
    tail_log_command,
    tail_log_in_terminal,
)
from rl_adaptive_dbs.tui.settings_data import (
    TuiSettings,
    parse_setting_input,
    save_settings,
    settings_file,
    settings_hints_line,
    settings_info_lines,
    settings_status_line,
    settings_table_rows,
    step_setting,
    update_setting,
)
from rl_adaptive_dbs.tui.training_data import (
    TrainingRun,
    cycle_training_run_id,
    discover_training_runs,
    return_sparkline_data,
    run_selector_rows,
    select_training_run,
    training_empty_message,
    training_metadata_lines,
    training_status_line,
)


class TailLog(Log):
    """Plain log tail viewer; keyboard scroll is routed from ``RlDbsTuiApp`` when open."""

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self.auto_scroll = self.is_vertical_scroll_end

    def _sync_scrollbars(self) -> None:
        """Keep scrollbar thumb in sync after content or layout changes."""
        self._scroll_update(self.virtual_size)
        if self.show_vertical_scrollbar:
            self.vertical_scrollbar.position = self.scroll_y
        if self.show_horizontal_scrollbar:
            self.horizontal_scrollbar.position = self.scroll_x

    def action_scroll_up(self) -> None:
        self.scroll_up(force=True, animate=False, immediate=True)
        self.auto_scroll = self.is_vertical_scroll_end

    def action_scroll_down(self) -> None:
        self.scroll_down(force=True, animate=False, immediate=True)
        self.auto_scroll = self.is_vertical_scroll_end

    def action_scroll_left(self) -> None:
        self.scroll_left(force=True, animate=False, immediate=True)

    def action_scroll_right(self) -> None:
        self.scroll_right(force=True, animate=False, immediate=True)

    def action_page_up(self) -> None:
        self.scroll_page_up(force=True, animate=False)
        self.auto_scroll = self.is_vertical_scroll_end

    def action_page_down(self) -> None:
        self.scroll_page_down(force=True, animate=False)
        self.auto_scroll = self.is_vertical_scroll_end


class TrainingPane(Static):
    """Training tab: run selector, progress, return sparkline."""

    DEFAULT_CSS = """
    TrainingPane {
        height: 1fr;
    }
    #training-status {
        height: 1;
        padding: 0 1;
    }
    #training-empty {
        padding: 1 2;
        color: $text-muted;
    }
    #training-detail {
        height: auto;
        padding: 0 1;
    }
    #training-progress {
        padding: 0 1;
        height: 1;
    }
    #training-sparkline {
        height: 3;
        padding: 0 1;
    }
    #training-meta {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    #training-run-table {
        height: 1fr;
        min-height: 5;
    }
    """

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        refresh_s: float = 1.0,
        sparkline_episodes: int = 40,
    ) -> None:
        super().__init__()
        self.artifacts_dir = artifacts_dir
        self.refresh_s = refresh_s
        self.sparkline_episodes = sparkline_episodes
        self._refresh_timer = None
        self._runs = discover_training_runs(artifacts_dir)
        self._active_run_id: str | None = self._runs[0].run_id if self._runs else None

    def compose(self) -> ComposeResult:
        yield Static("", id="training-status")
        with Vertical(id="training-body"):
            yield Static("", id="training-empty")
            yield DataTable(id="training-run-table", zebra_stripes=True)
            yield Static("", id="training-detail")
            yield ProgressBar(id="training-progress", show_eta=False)
            yield Sparkline([], id="training-sparkline")
            yield Static("", id="training-meta")

    def on_mount(self) -> None:
        table = self.query_one("#training-run-table", DataTable)
        table.add_columns(
            ("run", "run"),
            ("seed", "seed"),
            ("episodes", "episodes"),
            ("last_return", "last_return"),
        )
        table.cursor_type = "row"
        table.can_focus = True
        self._cached_run_ids: tuple[str, ...] = ()
        self._restart_refresh_timer()
        self.reload_data()

    def _restart_refresh_timer(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(self.refresh_s, self.reload_data)

    def apply_refresh_s(self, refresh_s: float) -> None:
        self.refresh_s = refresh_s
        self._restart_refresh_timer()

    def apply_sparkline_episodes(self, episodes: int) -> None:
        self.sparkline_episodes = episodes
        run = select_training_run(self._runs, self._active_run_id)
        if run is not None:
            self._update_panels(run)

    def on_show(self) -> None:
        if self._runs:
            self.query_one("#training-run-table", DataTable).focus()

    def _sync_cursor_to_active_run(self) -> None:
        table = self.query_one("#training-run-table", DataTable)
        for index, run_row in enumerate(self._runs):
            if run_row.run_id == self._active_run_id:
                table.move_cursor(row=index)
                return

    def _update_panels(self, run: TrainingRun) -> None:
        status = self.query_one("#training-status", Static)
        detail = self.query_one("#training-detail", Static)
        progress = self.query_one("#training-progress", ProgressBar)
        sparkline = self.query_one("#training-sparkline", Sparkline)
        meta = self.query_one("#training-meta", Static)

        status.update(training_status_line(run, self._runs))
        detail.update(
            f"episode return (last {min(len(run.episodes), self.sparkline_episodes)} episodes)"
            if run.episodes
            else "episode return"
        )
        progress.update(total=run.planned_episodes, progress=run.current_episode)
        sparkline.data = return_sparkline_data(run, max_points=self.sparkline_episodes)
        meta.update("\n".join(training_metadata_lines(run)))

    def reload_data(self) -> None:
        self._runs = discover_training_runs(self.artifacts_dir)
        if self._active_run_id is None and self._runs:
            self._active_run_id = self._runs[0].run_id
        run = select_training_run(self._runs, self._active_run_id)

        status = self.query_one("#training-status", Static)
        empty = self.query_one("#training-empty", Static)
        table = self.query_one("#training-run-table", DataTable)
        detail = self.query_one("#training-detail", Static)
        progress = self.query_one("#training-progress", ProgressBar)
        sparkline = self.query_one("#training-sparkline", Sparkline)
        meta = self.query_one("#training-meta", Static)

        if not self._runs:
            status.update(f"No training logs under {self.artifacts_dir}/")
            empty.update(training_empty_message(self.artifacts_dir))
            empty.display = True
            table.display = False
            detail.update("")
            progress.display = False
            sparkline.display = False
            meta.update("")
            self._cached_run_ids = ()
            return

        empty.display = False
        table.display = True
        progress.display = True
        sparkline.display = True

        if run is None:
            status.update(f"No training logs under {self.artifacts_dir}/")
            return

        new_ids = tuple(item.run_id for item in self._runs)
        table_has_focus = table.has_focus
        if new_ids != self._cached_run_ids:
            self._cached_run_ids = new_ids
            table.clear()
            for run_row in self._runs:
                label, seed, episodes, last = run_selector_rows([run_row])[0]
                table.add_row(label, seed, episodes, last, key=run_row.run_id)
            if not table_has_focus:
                self._sync_cursor_to_active_run()
        else:
            # Episode counts / last return may change while a run is in progress.
            for run_row in self._runs:
                _, seed, episodes, last = run_selector_rows([run_row])[0]
                table.update_cell(run_row.run_id, "seed", seed)
                table.update_cell(run_row.run_id, "episodes", episodes)
                table.update_cell(run_row.run_id, "last_return", last)

        self._update_panels(run)

    def action_prev_run(self) -> None:
        self._active_run_id = cycle_training_run_id(self._runs, self._active_run_id, -1)
        self.reload_data()
        self._sync_cursor_to_active_run()

    def action_next_run(self) -> None:
        self._active_run_id = cycle_training_run_id(self._runs, self._active_run_id, 1)
        self.reload_data()
        self._sync_cursor_to_active_run()

    def _select_run_from_table(self, row_key) -> None:
        if row_key is None:
            return
        run_id = str(row_key.value)
        if run_id == self._active_run_id:
            return
        self._active_run_id = run_id
        run = select_training_run(self._runs, run_id)
        if run is not None:
            self._update_panels(run)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "training-run-table":
            return
        self._select_run_from_table(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "training-run-table":
            return
        self._select_run_from_table(event.row_key)


class EvalPane(Static):
    """Eval tab: per-run metrics and P_beta timeseries sparkline."""

    DEFAULT_CSS = """
    EvalPane {
        height: 1fr;
    }
    #eval-status {
        height: 1;
        padding: 0 1;
    }
    #eval-warning {
        height: auto;
        padding: 0 1;
        color: $warning;
    }
    #eval-empty {
        padding: 1 2;
        color: $text-muted;
    }
    #eval-run-table {
        height: 1fr;
        min-height: 5;
    }
    #eval-summary {
        height: auto;
        padding: 0 1;
    }
    #eval-sparkline-label {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #eval-sparkline {
        height: 3;
        padding: 0 1;
    }
    """

    def __init__(self, results_dir: Path, *, refresh_s: float = 1.0) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.refresh_s = refresh_s
        self._suites, suite, self._runs = load_eval_context(results_dir)
        self._active_suite: str | None = suite.name if suite else None
        self._active_run_id: str | None = self._runs[0].run_id if self._runs else None

    def compose(self) -> ComposeResult:
        yield Static("", id="eval-status")
        with Vertical():
            yield Static("", id="eval-warning")
            yield Static("", id="eval-empty")
            yield DataTable(id="eval-run-table", zebra_stripes=True)
            yield Static("", id="eval-summary")
            yield Static("P_beta per step", id="eval-sparkline-label")
            yield Sparkline([], id="eval-sparkline")

    def on_mount(self) -> None:
        table = self.query_one("#eval-run-table", DataTable)
        table.add_columns(
            "controller",
            "variant",
            "seed",
            "p_beta_mu",
            "reward_sum",
            "run_id",
        )
        table.cursor_type = "row"
        table.can_focus = True
        self._refresh_timer = None
        self._restart_refresh_timer()
        self.reload_data()

    def _restart_refresh_timer(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(self.refresh_s, self.reload_data)

    def apply_refresh_s(self, refresh_s: float) -> None:
        self.refresh_s = refresh_s
        self._restart_refresh_timer()

    def on_show(self) -> None:
        if self._runs:
            self.query_one("#eval-run-table", DataTable).focus()
        self.reload_data()

    def open_run(self, suite_name: str | None, run_id: str) -> None:
        """Select a suite/run (e.g. from Benchmarks Enter)."""
        if suite_name:
            self._active_suite = suite_name
        self._active_run_id = run_id
        self.reload_data()
        self._sync_cursor_to_active_run()

    def _sync_cursor_to_active_run(self) -> None:
        table = self.query_one("#eval-run-table", DataTable)
        for index, run_row in enumerate(self._runs):
            if run_row.run_id == self._active_run_id:
                table.move_cursor(row=index)
                return

    def _update_panels(self, suite, run: EvalRun) -> None:
        status = self.query_one("#eval-status", Static)
        warning = self.query_one("#eval-warning", Static)
        summary = self.query_one("#eval-summary", Static)
        sparkline_label = self.query_one("#eval-sparkline-label", Static)
        sparkline = self.query_one("#eval-sparkline", Sparkline)

        status.update(eval_status_line(suite, run, self._runs))
        banner = cross_paper_warning(suite)
        if banner:
            warning.update(banner)
            warning.display = True
        else:
            warning.update("")
            warning.display = False
        summary.update("\n".join(eval_summary_lines(run, show_reward=suite.show_reward_sum)))
        if run.has_timeseries:
            points = len(run.timeseries.get("p_beta", [])) if run.timeseries else 0
            shown = min(points, 10_000)
            sparkline_label.update(f"P_beta per step (last {shown} of {points})")
            sparkline.data = p_beta_sparkline_data(run)
            sparkline_label.display = True
            sparkline.display = True
        else:
            sparkline_label.update("P_beta per step — no timeseries/ for this run")
            sparkline.data = []
            sparkline_label.display = True
            sparkline.display = False

    def reload_data(self) -> None:
        self._suites, suite, self._runs = load_eval_context(
            self.results_dir,
            suite_name=self._active_suite,
        )
        if self._active_suite is None and suite is not None:
            self._active_suite = suite.name
        if self._active_run_id is None and self._runs:
            self._active_run_id = self._runs[0].run_id

        status = self.query_one("#eval-status", Static)
        warning = self.query_one("#eval-warning", Static)
        empty = self.query_one("#eval-empty", Static)
        table = self.query_one("#eval-run-table", DataTable)
        summary = self.query_one("#eval-summary", Static)
        sparkline_label = self.query_one("#eval-sparkline-label", Static)
        sparkline = self.query_one("#eval-sparkline", Sparkline)

        if suite is None or not self._runs:
            status.update(f"No benchmark suites under {self.results_dir}/")
            empty.update(eval_empty_message(self.results_dir))
            empty.display = True
            warning.display = False
            table.display = False
            summary.update("")
            sparkline_label.display = False
            sparkline.display = False
            return

        empty.display = False
        table.display = True

        run = select_eval_run(self._runs, self._active_run_id)
        if run is None:
            status.update(f"No runs in {suite.name}")
            return

        self._active_run_id = run.run_id
        table_has_focus = table.has_focus
        table.clear()
        for index, row_cells in enumerate(
            eval_run_table_rows(self._runs, show_reward=suite.show_reward_sum)
        ):
            table.add_row(*row_cells, key=self._runs[index].run_id)
        if not table_has_focus:
            self._sync_cursor_to_active_run()

        self._update_panels(suite, run)

    def action_prev_run(self) -> None:
        self._active_run_id = cycle_eval_run_id(self._runs, self._active_run_id, -1)
        self.reload_data()
        self._sync_cursor_to_active_run()

    def action_next_run(self) -> None:
        self._active_run_id = cycle_eval_run_id(self._runs, self._active_run_id, 1)
        self.reload_data()
        self._sync_cursor_to_active_run()

    def _select_run_from_table(self, row_key) -> None:
        if row_key is None:
            return
        run_id = str(row_key.value)
        if run_id == self._active_run_id:
            return
        self._active_run_id = run_id
        self._suites, suite, self._runs = load_eval_context(
            self.results_dir,
            suite_name=self._active_suite,
        )
        run = select_eval_run(self._runs, run_id)
        if run is not None and suite is not None:
            self._update_panels(suite, run)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "eval-run-table":
            return
        self._select_run_from_table(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "eval-run-table":
            return
        self._select_run_from_table(event.row_key)


class BenchmarksPane(Static):
    """Benchmarks tab: suite selector + comparison table."""

    DEFAULT_CSS = """
    BenchmarksPane {
        height: 1fr;
    }
    #suite-status {
        height: 1;
        padding: 0 1;
    }
    #filter-input {
        dock: top;
        display: none;
    }
    #filter-input.visible {
        display: block;
    }
    #benchmark-table {
        height: 1fr;
    }
    """

    def __init__(self, results_dir: Path, *, refresh_s: float = 1.0) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.refresh_s = refresh_s
        self._suites = refresh_suites(results_dir)
        self._active_suite: str | None = self._suites[0].name if self._suites else None
        self._filter = ""
        self._highlighted_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Input(placeholder="filter controller or variant…", id="filter-input")
        yield Static("No results/", id="suite-status")
        yield DataTable(id="benchmark-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#benchmark-table", DataTable)
        table.add_columns(
            "controller",
            "variant",
            "seed",
            "p_beta_mu",
            "p_beta_fin",
            "reward_sum",
            "run_id",
        )
        table.cursor_type = "row"
        table.can_focus = True
        self._refresh_timer = None
        self._restart_refresh_timer()
        self.reload_data()

    def _restart_refresh_timer(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(self.refresh_s, self.reload_data)

    def apply_refresh_s(self, refresh_s: float) -> None:
        self.refresh_s = refresh_s
        self._restart_refresh_timer()

    def on_show(self) -> None:
        if self._suites:
            self.query_one("#benchmark-table", DataTable).focus()

    def reload_data(self) -> None:
        self._suites = refresh_suites(self.results_dir)
        if self._active_suite is None and self._suites:
            self._active_suite = self._suites[0].name
        suite = select_suite(self._suites, self._active_suite)
        status = self.query_one("#suite-status", Static)
        table = self.query_one("#benchmark-table", DataTable)
        table.clear()
        if suite is None:
            status.update(f"No benchmark suites under {self.results_dir}/")
            return
        status.update(suite_status_line(suite))
        filtered = filter_runs(suite.runs, self._filter)
        for run in filtered:
            if run.error:
                cells = (
                    run.controller,
                    run.variant,
                    str(run.seed),
                    "ERR",
                    "ERR",
                    run.error[:24],
                    run.run_id[:20],
                )
            else:
                reward = f"{run.reward_sum:.1f}" if suite.show_reward_sum else "n/a"
                cells = (
                    run.controller,
                    run.variant,
                    str(run.seed),
                    f"{run.p_beta_mean:.1f}",
                    f"{run.p_beta_final:.1f}",
                    reward,
                    run.run_id[:20],
                )
            table.add_row(*cells, key=run.run_id)
        if self._highlighted_run_id is None and filtered:
            self._highlighted_run_id = filtered[0].run_id

    def highlighted_run_id(self) -> str | None:
        return self._highlighted_run_id

    def active_suite_name(self) -> str | None:
        return self._active_suite

    def _select_run_from_table(self, row_key) -> None:
        if row_key is None:
            return
        self._highlighted_run_id = str(row_key.value)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "benchmark-table":
            return
        self._select_run_from_table(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "benchmark-table":
            return
        self._select_run_from_table(event.row_key)

    def action_open_filter(self) -> None:
        field = self.query_one("#filter-input", Input)
        field.add_class("visible")
        field.focus()

    def action_clear_filter(self) -> None:
        self._filter = ""
        field = self.query_one("#filter-input", Input)
        field.value = ""
        field.remove_class("visible")
        self.reload_data()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "filter-input":
            return
        self._filter = event.value
        event.input.remove_class("visible")
        self.reload_data()



class LogsPane(Static):
    """Logs tab: full-width file list, or full-screen tail view when a log is open."""

    DEFAULT_CSS = """
    LogsPane {
        height: 1fr;
    }
    #logs-status {
        height: 1;
        padding: 0 1;
    }
    #logs-empty {
        padding: 1 2;
        color: $text-muted;
    }
    #logs-filter-input {
        dock: top;
        display: none;
    }
    #logs-filter-input.visible {
        display: block;
    }
    #logs-body {
        height: 1fr;
    }
    #logs-split {
        height: 1fr;
    }
    #logs-file-table {
        width: 1fr;
        height: 1fr;
        min-height: 5;
    }
    #logs-view {
        width: 1fr;
        height: 1fr;
        display: none;
    }
    #logs-view.visible {
        display: block;
    }
    #logs-view:focus {
        border: solid $accent;
    }
    #logs-hints {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        results_dir: Path,
        artifacts_dir: Path,
        *,
        logs_dir: Path,
        refresh_s: float = 1.0,
        color_enabled: bool = False,
        tail_lines: int = 200,
    ) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self.refresh_s = refresh_s
        self.color_enabled = color_enabled
        self.tail_lines = tail_lines
        self._bookmarks_path = bookmarks_file(artifacts_dir)
        self._files = discover_log_files(results_dir, artifacts_dir, logs_dir=logs_dir)
        self._highlighted_path: Path | None = None
        self._opened_path: Path | None = None
        self._filter = ""
        self._cached_file_list_key: tuple[str, ...] = ()
        self._cached_file_stats: tuple[tuple[str, float, int], ...] = ()
        self._cached_tail_key: tuple[str, float, int] | None = None

    def compose(self) -> ComposeResult:
        yield Input(placeholder="filter path or source…", id="logs-filter-input")
        yield Static("", id="logs-status")
        with Vertical(id="logs-body"):
            yield Static("", id="logs-empty")
            with Horizontal(id="logs-split"):
                yield DataTable(id="logs-file-table", zebra_stripes=True)
                yield TailLog(
                    id="logs-view",
                    highlight=False,
                    auto_scroll=False,
                )
        yield Static("", id="logs-hints")

    def on_mount(self) -> None:
        table = self.query_one("#logs-file-table", DataTable)
        table.add_columns(
            ("source", "source"),
            ("path", "path"),
            ("size", "size"),
            ("run", "run"),
        )
        table.cursor_type = "row"
        table.can_focus = True
        self._apply_layout_mode()
        self._refresh_timer = None
        self._restart_refresh_timer()
        self.reload_data()
        self._update_hints()

    def _restart_refresh_timer(self) -> None:
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self.set_interval(self.refresh_s, self.reload_data)

    def apply_refresh_s(self, refresh_s: float) -> None:
        self.refresh_s = refresh_s
        self._restart_refresh_timer()

    def on_show(self) -> None:
        if self._opened_path is not None:
            self._viewer().focus()
        elif self._files:
            self.query_one("#logs-file-table", DataTable).focus()

    def viewer_has_focus(self) -> bool:
        return self.is_viewing_log() and self._viewer().has_focus

    def is_viewing_log(self) -> bool:
        return self._opened_path is not None

    def focus_file_list(self) -> None:
        self.query_one("#logs-file-table", DataTable).focus()

    def _viewer(self) -> TailLog:
        return self.query_one("#logs-view", TailLog)

    def focus_viewer(self) -> None:
        if self.is_viewing_log():
            self._viewer().focus()

    def _apply_layout_mode(self) -> None:
        browsing = self._opened_path is None
        table = self.query_one("#logs-file-table", DataTable)
        viewer = self.query_one("#logs-view", TailLog)
        table.display = browsing
        table.disabled = not browsing
        if browsing:
            viewer.remove_class("visible")
        else:
            viewer.add_class("visible")
            self.call_after_refresh(self.focus_viewer)

    def on_resize(self, event) -> None:
        if self.is_viewing_log():
            self.call_after_refresh(self._viewer()._sync_scrollbars)

    def _filtered_files(self) -> list[LogFile]:
        return filter_log_files(self._files, self._filter)

    def _update_hints(self) -> None:
        self.query_one("#logs-hints", Static).update(
            logs_hints_line(viewing=self.is_viewing_log())
        )

    def _update_status(self) -> None:
        visible = self._filtered_files()
        opened = select_log_file(visible, self._opened_path) if self._opened_path else None
        highlighted = select_log_file(visible, self._highlighted_path) if self._highlighted_path else None
        self.query_one("#logs-status", Static).update(
            logs_status_line(
                visible,
                opened,
                highlighted=highlighted,
                tail_lines=self.tail_lines,
            )
        )
        self._update_hints()

    def action_back_to_list(self) -> None:
        if self._opened_path is None:
            self.focus_file_list()
            return
        self._opened_path = None
        self._apply_layout_mode()
        self._update_status()
        self.focus_file_list()

    def _sync_cursor_to_highlighted(self) -> None:
        table = self.query_one("#logs-file-table", DataTable)
        visible = self._filtered_files()
        target = self._highlighted_path
        if target is None and visible:
            target = visible[0].path
        for index, item in enumerate(visible):
            if item.path == target:
                table.move_cursor(row=index)
                return

    def _render_tail(self, log_file: LogFile, *, force: bool = False) -> None:
        viewer = self.query_one("#logs-view", TailLog)
        tail_key = (str(log_file.path), log_file.mtime, log_file.size)
        if not force and tail_key == self._cached_tail_key:
            return
        if not force and not viewer.is_vertical_scroll_end:
            return

        self._cached_tail_key = tail_key
        follow_tail = viewer.is_vertical_scroll_end or force
        viewer.clear()
        if log_file.size == 0 and not log_file.path.is_file():
            viewer.write(f"missing: {log_file.display_path}\n", scroll_end=False)
            return

        lines, error = tail_lines(log_file.path, max_lines=self.tail_lines)
        if error is not None:
            viewer.write(f"{error}\n", scroll_end=False)
            return
        saved_auto_scroll = viewer.auto_scroll
        viewer.auto_scroll = False
        for line in lines:
            viewer.write(f"{line}\n", scroll_end=False)
        viewer.auto_scroll = saved_auto_scroll
        if follow_tail:
            viewer.scroll_end(animate=False, immediate=True)
        viewer._sync_scrollbars()

    def reload_data(self) -> None:
        self._files = discover_log_files(
            self.results_dir,
            self.artifacts_dir,
            logs_dir=self.logs_dir,
        )
        visible = self._filtered_files()
        if self._highlighted_path is None and visible:
            self._highlighted_path = visible[0].path

        status = self.query_one("#logs-status", Static)
        empty = self.query_one("#logs-empty", Static)
        table = self.query_one("#logs-file-table", DataTable)
        viewer = self.query_one("#logs-view", TailLog)

        if not visible:
            status.update(
                f"No log files under {self.results_dir}/, {self.artifacts_dir}/, or {self.logs_dir}/"
                if not self._filter
                else f"No log files match filter: {self._filter!r}"
            )
            empty.update(
                logs_empty_message(self.results_dir, self.artifacts_dir, self.logs_dir)
            )
            empty.display = not self._filter
            table.display = bool(visible)
            self._opened_path = None
            self._highlighted_path = None
            viewer.clear()
            self._cached_file_list_key = ()
            self._cached_file_stats = ()
            self._cached_tail_key = None
            return

        empty.display = False

        if self._opened_path is not None:
            opened = select_log_file(visible, self._opened_path)
            if opened is None:
                self._opened_path = None
                self._apply_layout_mode()
            elif viewer.has_class("visible"):
                self._render_tail(opened)
        else:
            self._apply_layout_mode()

        table.display = self._opened_path is None
        if table.display:
            list_key = tuple(log_row_key(item) for item in visible)
            stats_key = tuple(
                (log_row_key(item), item.mtime, item.size, item.run_state) for item in visible
            )
            table_has_focus = table.has_focus
            if list_key != self._cached_file_list_key:
                self._cached_file_list_key = list_key
                self._cached_file_stats = stats_key
                table.clear()
                for index, row in enumerate(log_file_rows(visible)):
                    source, path_label, size, run_label = row
                    table.add_row(source, path_label, size, run_label, key=log_row_key(visible[index]))
                if not table_has_focus:
                    self._sync_cursor_to_highlighted()
            elif stats_key != self._cached_file_stats:
                for item in visible:
                    _, _, size, run_label = log_file_rows([item])[0]
                    key = log_row_key(item)
                    table.update_cell(key, "size", size)
                    table.update_cell(key, "run", run_label)
                self._cached_file_stats = stats_key
        else:
            self._cached_file_list_key = ()
            self._cached_file_stats = ()

        self._update_status()

    def force_refresh_opened_log(self) -> None:
        if self._opened_path is None:
            return
        opened = select_log_file(self._filtered_files(), self._opened_path)
        if opened is not None:
            self._cached_tail_key = None
            self._render_tail(opened, force=True)

    def apply_tail_lines(self, tail_lines: int) -> None:
        self.tail_lines = tail_lines
        self._update_status()
        self.force_refresh_opened_log()

    def action_open_filter(self) -> None:
        field = self.query_one("#logs-filter-input", Input)
        field.add_class("visible")
        field.focus()

    def action_clear_filter(self) -> None:
        self._filter = ""
        field = self.query_one("#logs-filter-input", Input)
        field.value = ""
        field.remove_class("visible")
        self.reload_data()

    def action_bookmark_active(self) -> None:
        path = self._highlighted_path or self._opened_path
        if path is None:
            self.notify("No log file selected", title="Logs", timeout=3)
            return
        bookmarked = toggle_bookmark(self._bookmarks_path, path)
        verb = "Bookmarked" if bookmarked else "Removed bookmark for"
        self.notify(f"{verb} {path.name}", title="Logs", timeout=3)
        self._cached_file_list_key = ()
        self.reload_data()

    def action_prev_file(self) -> None:
        visible = self._filtered_files()
        self._highlighted_path = cycle_log_file(visible, self._highlighted_path, -1)
        self._sync_cursor_to_highlighted()
        self._update_status()

    def action_next_file(self) -> None:
        visible = self._filtered_files()
        self._highlighted_path = cycle_log_file(visible, self._highlighted_path, 1)
        self._sync_cursor_to_highlighted()
        self._update_status()

    def _open_file(self, path: Path) -> None:
        self._opened_path = path
        self._highlighted_path = path
        self._cached_tail_key = None
        active = select_log_file(self._filtered_files(), path)
        if active is None:
            return
        self._apply_layout_mode()
        self._update_status()
        self._render_tail(active, force=True)
        self._viewer().auto_scroll = True
        self.focus_viewer()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "logs-file-table":
            return
        if event.row_key is None:
            return
        self._highlighted_path = Path(str(event.row_key.value))
        self._update_status()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "logs-file-table":
            return
        if event.row_key is None:
            return
        self._open_file(Path(str(event.row_key.value)))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "logs-filter-input":
            return
        self._filter = event.value
        event.input.remove_class("visible")
        self._cached_file_list_key = ()
        self._cached_file_stats = ()
        self._cached_tail_key = None
        self.reload_data()


class SettingsDataTable(DataTable):
    """Settings table that handles ``+`` / ``-`` even when key names vary by terminal."""

    def on_key(self, event: events.Key) -> None:
        pane = self.parent
        if not isinstance(pane, SettingsPane):
            return
        if pane.query_one("#settings-edit-input", Input).has_class("visible"):
            return
        if event.character == "-" or event.key in {"minus", "hyphen_minus"}:
            pane.action_adjust_setting(-1)
            event.prevent_default()
            event.stop()
            return
        if event.character == "+" or event.key in {
            "plus",
            "shift+equal",
            "shift+equals_sign",
            "equals_sign",
        }:
            pane.action_adjust_setting(1)
            event.prevent_default()
            event.stop()
            return
        if event.key == "space":
            pane.action_toggle_selected()
            event.prevent_default()
            event.stop()


class SettingsPane(Static):
    """Settings tab: edit persisted TUI preferences."""

    DEFAULT_CSS = """
    SettingsPane {
        height: 1fr;
    }
    #settings-status {
        height: 1;
        padding: 0 1;
    }
    #settings-edit-input {
        dock: top;
        display: none;
    }
    #settings-edit-input.visible {
        display: block;
    }
    #settings-table {
        height: auto;
        min-height: 6;
        max-height: 8;
    }
    #settings-info {
        height: 1fr;
        padding: 0 1;
        color: $text-muted;
    }
    #settings-hints {
        dock: bottom;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        results_dir: Path,
        artifacts_dir: Path,
        settings: TuiSettings,
        *,
        logs_dir: Path,
    ) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self._settings_path = settings_file(artifacts_dir)
        self._bookmarks_path = bookmarks_file(artifacts_dir)
        self.settings = settings
        self._selected_key: str | None = settings_table_rows(settings)[0][0]

    def compose(self) -> ComposeResult:
        yield Input(placeholder="new value…", id="settings-edit-input")
        yield Static("", id="settings-status")
        yield SettingsDataTable(id="settings-table", zebra_stripes=True)
        yield Static("", id="settings-info")
        yield Static("", id="settings-hints")

    def on_mount(self) -> None:
        table = self.query_one("#settings-table", SettingsDataTable)
        table.add_columns("setting", "value")
        table.cursor_type = "row"
        table.can_focus = True
        self.reload_data()

    def on_show(self) -> None:
        self.query_one("#settings-table", SettingsDataTable).focus()
        self._sync_cursor_to_selected()

    def _sync_cursor_to_selected(self) -> None:
        table = self.query_one("#settings-table", SettingsDataTable)
        rows = settings_table_rows(self.settings)
        for index, (key, _, _) in enumerate(rows):
            if key == self._selected_key:
                table.move_cursor(row=index)
                return

    def _update_status(self) -> None:
        self.query_one("#settings-status", Static).update(
            settings_status_line(self.settings, selected_key=self._selected_key)
        )

    def _update_info(self) -> None:
        self.query_one("#settings-info", Static).update(
            "\n".join(
                settings_info_lines(
                    results_dir=self.results_dir,
                    artifacts_dir=self.artifacts_dir,
                    logs_dir=self.logs_dir,
                    settings_path=self._settings_path,
                    bookmarks_path=self._bookmarks_path,
                )
            )
        )
        self.query_one("#settings-hints", Static).update(settings_hints_line())

    def reload_data(self) -> None:
        table = self.query_one("#settings-table", SettingsDataTable)
        rows = settings_table_rows(self.settings)
        if self._selected_key is None and rows:
            self._selected_key = rows[0][0]
        table.clear()
        for key, label, value in rows:
            table.add_row(label, value, key=key)
        self._sync_cursor_to_selected()
        self._update_status()
        self._update_info()

    def _commit_settings(self, settings: TuiSettings) -> None:
        self.settings = settings
        save_settings(self._settings_path, settings)
        self.reload_data()
        app = self.app
        if hasattr(app, "apply_tui_settings"):
            app.apply_tui_settings(settings)

    def highlighted_setting_key(self) -> str | None:
        return self._selected_key

    def action_open_editor(self) -> None:
        key = self.highlighted_setting_key()
        if key is None:
            return
        current = next(value for row_key, _, value in settings_table_rows(self.settings) if row_key == key)
        field = self.query_one("#settings-edit-input", Input)
        field.value = current
        field.add_class("visible")
        field.focus()

    def action_adjust_setting(self, delta: int) -> None:
        key = self.highlighted_setting_key()
        if key is None:
            return
        self._commit_settings(step_setting(self.settings, key, delta))

    def action_toggle_selected(self) -> None:
        key = self.highlighted_setting_key()
        if key == "color_enabled":
            self._commit_settings(step_setting(self.settings, key, 1))
            return
        if key == "launch_follow":
            self._commit_settings(step_setting(self.settings, key, 1))

    def action_toggle_color(self) -> None:
        self.action_toggle_selected()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "settings-table":
            return
        if event.row_key is None:
            return
        self._selected_key = str(event.row_key.value)
        self._update_status()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "settings-table":
            return
        if event.row_key is not None:
            self._selected_key = str(event.row_key.value)
        self.action_open_editor()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "settings-edit-input":
            return
        key = self.highlighted_setting_key()
        field = self.query_one("#settings-edit-input", Input)
        field.remove_class("visible")
        if key is None:
            return
        parsed = parse_setting_input(key, event.value)
        if parsed is None:
            self.notify("Invalid value", title="Settings", severity="error", timeout=3)
            self.query_one("#settings-table", SettingsDataTable).focus()
            return
        self._commit_settings(update_setting(self.settings, key, parsed))
        self.query_one("#settings-table", SettingsDataTable).focus()


class ConfirmLaunchScreen(ModalScreen[bool]):
    """Confirm before starting a detached run."""

    BINDINGS = [
        Binding("f", "cycle_follow", "Follow mode", show=False),
    ]

    DEFAULT_CSS = """
    ConfirmLaunchScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 80;
        height: auto;
        max-width: 90%;
        border: thick $accent;
        padding: 1 2;
        background: $surface;
    }
    #confirm-command {
        color: $text-muted;
        padding: 1 0;
    }
    #confirm-follow {
        color: $text-muted;
        padding: 1 0 0 0;
    }
    #confirm-buttons {
        height: auto;
        padding-top: 1;
    }
    """

    def __init__(self, recipe: RunRecipe, *, initial_follow: str) -> None:
        super().__init__()
        self.recipe = recipe
        self.follow_mode = initial_follow

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(f"Launch detached?\n{self.recipe.label}")
            yield Static(self.recipe.command_preview, id="confirm-command")
            yield Static("", id="confirm-follow")
            with Horizontal(id="confirm-buttons"):
                yield Button("Launch", variant="primary", id="launch")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self._update_follow_label()

    def _update_follow_label(self) -> None:
        self.query_one("#confirm-follow", Static).update(
            f"Follow output: {launch_follow_label(self.follow_mode)}  (f: cycle)"
        )

    def action_cycle_follow(self) -> None:
        self.follow_mode = cycle_dialog_follow(self.follow_mode)
        self._update_follow_label()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "launch")


class RunPane(Static):
    """Run tab: browse recipes and launch detached jobs."""

    DEFAULT_CSS = """
    RunPane {
        height: 1fr;
    }
    #run-status {
        height: 1;
        padding: 0 1;
    }
    #run-empty {
        padding: 1 2;
        color: $text-muted;
    }
    #run-filter-input {
        dock: top;
        display: none;
    }
    #run-filter-input.visible {
        display: block;
    }
    #run-recipe-table {
        height: 1fr;
        min-height: 5;
    }
    #run-detail {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        artifacts_dir: Path,
        *,
        repo_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.artifacts_dir = artifacts_dir
        self.repo_root = (repo_root or find_repo_root()).resolve()
        self._recipes = discover_run_recipes(self.repo_root)
        self._active_recipe_id: str | None = self._recipes[0].recipe_id if self._recipes else None
        self._filter = ""
        self._last_launch: LaunchResult | None = None
        self._last_launch_label: str | None = None

    def compose(self) -> ComposeResult:
        yield Input(placeholder="filter label, category, or command…", id="run-filter-input")
        yield Static("", id="run-status")
        with Vertical():
            yield Static("", id="run-empty")
            yield DataTable(id="run-recipe-table", zebra_stripes=True)
            yield Static("", id="run-detail")

    def on_mount(self) -> None:
        table = self.query_one("#run-recipe-table", DataTable)
        table.add_columns("category", "label", "command")
        table.cursor_type = "row"
        table.can_focus = True
        self.reload_data()

    def on_show(self) -> None:
        if self._visible_recipes():
            self.query_one("#run-recipe-table", DataTable).focus()

    def _visible_recipes(self) -> list[RunRecipe]:
        return filter_recipes(self._recipes, self._filter)

    def active_recipe(self) -> RunRecipe | None:
        return select_recipe(self._visible_recipes(), self._active_recipe_id)

    def last_launch_log(self) -> Path | None:
        return self._last_launch.log_path if self._last_launch else None

    def reload_data(self) -> None:
        self._recipes = discover_run_recipes(self.repo_root)
        visible = self._visible_recipes()
        if self._active_recipe_id is None and visible:
            self._active_recipe_id = visible[0].recipe_id

        status = self.query_one("#run-status", Static)
        empty = self.query_one("#run-empty", Static)
        table = self.query_one("#run-recipe-table", DataTable)
        detail = self.query_one("#run-detail", Static)

        if not self._recipes:
            status.update(run_status_line([], None))
            empty.update(run_empty_message(self.repo_root))
            empty.display = True
            table.display = False
            detail.update("")
            return

        empty.display = False
        table.display = True
        active = self.active_recipe()
        status.update(
            run_status_line(visible, active, last_launch=self._last_launch_label)
        )

        table.clear()
        for index, row in enumerate(recipe_table_rows(visible)):
            table.add_row(*row, key=visible[index].recipe_id)
        self._sync_cursor_to_active()
        self._update_detail(active)

    def _update_detail(self, recipe: RunRecipe | None) -> None:
        detail = self.query_one("#run-detail", Static)
        if recipe is None:
            detail.update("")
            return
        lines = [recipe.command_preview]
        if recipe.description:
            lines.append(recipe.description)
        lines.append(
            "Enter or x: launch detached  |  f: follow mode in confirm dialog"
            "  |  output → artifacts/tui-runs/"
        )
        detail.update("\n".join(lines))

    def _sync_cursor_to_active(self) -> None:
        table = self.query_one("#run-recipe-table", DataTable)
        visible = self._visible_recipes()
        for index, item in enumerate(visible):
            if item.recipe_id == self._active_recipe_id:
                table.move_cursor(row=index)
                return

    def _select_recipe_from_table(self, row_key) -> None:
        if row_key is None:
            return
        recipe_id = str(row_key.value)
        if recipe_id == self._active_recipe_id:
            return
        self._active_recipe_id = recipe_id
        self._update_detail(self.active_recipe())
        status = self.query_one("#run-status", Static)
        status.update(
            run_status_line(
                self._visible_recipes(),
                self.active_recipe(),
                last_launch=self._last_launch_label,
            )
        )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "run-recipe-table":
            return
        self._select_recipe_from_table(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "run-recipe-table":
            return
        self._select_recipe_from_table(event.row_key)

    def action_open_filter(self) -> None:
        field = self.query_one("#run-filter-input", Input)
        field.add_class("visible")
        field.focus()

    def action_clear_filter(self) -> None:
        self._filter = ""
        field = self.query_one("#run-filter-input", Input)
        field.value = ""
        field.remove_class("visible")
        self.reload_data()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "run-filter-input":
            return
        self._filter = event.value
        event.input.remove_class("visible")
        self.reload_data()

    def action_prev_recipe(self) -> None:
        visible = self._visible_recipes()
        self._active_recipe_id = cycle_recipe_id(visible, self._active_recipe_id, -1)
        self.reload_data()

    def action_next_recipe(self) -> None:
        visible = self._visible_recipes()
        self._active_recipe_id = cycle_recipe_id(visible, self._active_recipe_id, 1)
        self.reload_data()

    async def action_confirm_launch(self) -> None:
        recipe = self.active_recipe()
        if recipe is None:
            self.notify("No recipe selected", title="Run", timeout=3)
            return
        app = self.app
        settings = app.settings if isinstance(app, RlDbsTuiApp) else TuiSettings()
        initial_follow = initial_dialog_follow(settings.launch_follow)
        screen = ConfirmLaunchScreen(recipe, initial_follow=initial_follow)
        confirmed = await self.app.push_screen_wait(screen)
        if not confirmed:
            return
        try:
            result = launch_detached(
                list(recipe.argv),
                repo_root=self.repo_root,
                artifacts_dir=self.artifacts_dir,
                recipe_id=recipe.log_recipe_id(),
            )
        except (OSError, ValueError) as exc:
            self.notify(str(exc), title="Launch failed", severity="error", timeout=6)
            return
        self._last_launch = result
        self._last_launch_label = f"started pid {result.pid} → {result.log_path.name}"
        self.notify(
            f"pid {result.pid}  log: {result.log_path.name}",
            title="Launched",
            timeout=5,
        )
        self.reload_data()
        if isinstance(app, RlDbsTuiApp):
            app.follow_launch_log(result.log_path, screen.follow_mode)


class RlDbsTuiApp(App):
    """Monitor and launch — Training, Eval, Benchmarks, Run, Logs, and Settings tabs."""

    TITLE = "rl-dbs-tui"
    CSS = """
    Header > HeaderIcon {
        display: none;
    }
    TabbedContent {
        height: 1fr;
    }
    TabbedContent Tabs {
        height: 1;
    }
    TabbedContent Tabs Tab.-active {
        background: $block-cursor-background;
        color: $block-cursor-foreground;
        text-style: $block-cursor-text-style;
    }
    TabbedContent Underline {
        display: none;
    }
    .placeholder {
        padding: 1 2;
        color: $text-muted;
    }
    """

    TAB_ORDER = (
        "tab-run",
        "tab-training",
        "tab-eval",
        "tab-benchmarks",
        "tab-logs",
        "tab-settings",
    )

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+r", "restart", "Restart"),
        Binding("/", "filter", "Filter", show=True),
        Binding("escape", "clear_filter", "Clear filter"),
        Binding("left", "prev_tab", "Prev tab", show=False, priority=True),
        Binding("right", "next_tab", "Next tab", show=False, priority=True),
        Binding("up", "logs_scroll_up", "Up", show=False, priority=True),
        Binding("down", "logs_scroll_down", "Down", show=False, priority=True),
        Binding("pageup", "logs_page_up", "Page up", show=False, priority=True),
        Binding("pagedown", "logs_page_down", "Page down", show=False, priority=True),
        Binding("tab", "next_tab", "Next tab", show=False, priority=True),
        Binding("shift+tab", "prev_tab", "Prev tab", show=False, priority=True),
        Binding("1", "jump_tab_1", "Run", show=False),
        Binding("2", "jump_tab_2", "Training", show=False),
        Binding("3", "jump_tab_3", "Eval", show=False),
        Binding("4", "jump_tab_4", "Benchmarks", show=False),
        Binding("5", "jump_tab_5", "Logs", show=False),
        Binding("6", "jump_tab_6", "Settings", show=False),
        Binding("b", "bookmark", "Toggle bookmark", show=True),
        Binding("x", "launch", "Launch", show=False),
        Binding("+", "increase_setting", "Increase", show=False),
        Binding("shift+equal", "increase_setting", "Increase", show=False),
        Binding("plus", "increase_setting", "Increase", show=False),
        Binding("-", "decrease_setting", "Decrease", show=False),
        Binding("minus", "decrease_setting", "Decrease", show=False),
        Binding("space", "toggle_setting", "Toggle", show=False),
        Binding("[", "prev_item", "Prev", show=False),
        Binding("]", "next_item", "Next", show=False),
        Binding("enter", "open_detail", "Detail", show=False),
        Binding("question_mark", "help", "Help", show=False),
    ]

    def __init__(
        self,
        results_dir: Path,
        *,
        artifacts_dir: Path,
        logs_dir: Path,
        settings: TuiSettings,
    ) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.artifacts_dir = artifacts_dir
        self.logs_dir = logs_dir
        self.settings = settings
        self.refresh_s = settings.refresh_s
        self.color_enabled = settings.color_enabled

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Run", id="tab-run"):
                yield RunPane(self.artifacts_dir)
            with TabPane("Training", id="tab-training"):
                yield TrainingPane(
                    self.artifacts_dir,
                    refresh_s=self.settings.refresh_s,
                    sparkline_episodes=self.settings.sparkline_episodes,
                )
            with TabPane("Eval", id="tab-eval"):
                yield EvalPane(self.results_dir, refresh_s=self.settings.refresh_s)
            with TabPane("Benchmarks", id="tab-benchmarks"):
                yield BenchmarksPane(self.results_dir, refresh_s=self.settings.refresh_s)
            with TabPane("Logs", id="tab-logs"):
                yield LogsPane(
                    self.results_dir,
                    self.artifacts_dir,
                    logs_dir=self.logs_dir,
                    refresh_s=self.settings.refresh_s,
                    color_enabled=self.settings.color_enabled,
                    tail_lines=self.settings.tail_lines,
                )
            with TabPane("Settings", id="tab-settings"):
                yield SettingsPane(
                    self.results_dir,
                    self.artifacts_dir,
                    self.settings,
                    logs_dir=self.logs_dir,
                )
        yield Footer()

    def apply_tui_settings(self, settings: TuiSettings) -> None:
        self.settings = settings
        self.refresh_s = settings.refresh_s
        self.color_enabled = settings.color_enabled
        self._training_pane().apply_refresh_s(settings.refresh_s)
        self._training_pane().apply_sparkline_episodes(settings.sparkline_episodes)
        self._eval_pane().apply_refresh_s(settings.refresh_s)
        self._benchmarks_pane().apply_refresh_s(settings.refresh_s)
        self._logs_pane().apply_refresh_s(settings.refresh_s)
        self._logs_pane().apply_tail_lines(settings.tail_lines)

    def _tabbed_content(self) -> TabbedContent:
        return self.query_one(TabbedContent)

    def _cycle_tab(self, delta: int) -> None:
        tabs = self._tabbed_content()
        order = self.TAB_ORDER
        current = tabs.active
        if current not in order:
            tabs.active = order[0]
            return
        tabs.active = order[(order.index(current) + delta) % len(order)]

    def action_prev_tab(self) -> None:
        if self._active_tab_id() == "tab-logs" and self._logs_pane().is_viewing_log():
            self._logs_pane()._viewer().action_scroll_left()
            return
        self._cycle_tab(-1)

    def action_next_tab(self) -> None:
        if self._active_tab_id() == "tab-logs" and self._logs_pane().is_viewing_log():
            self._logs_pane()._viewer().action_scroll_right()
            return
        self._cycle_tab(1)

    def _logs_viewer_action(self, name: str) -> None:
        if self._active_tab_id() != "tab-logs" or not self._logs_pane().is_viewing_log():
            raise SkipAction()
        getattr(self._logs_pane()._viewer(), f"action_{name}")()

    def action_logs_scroll_up(self) -> None:
        self._logs_viewer_action("scroll_up")

    def action_logs_scroll_down(self) -> None:
        self._logs_viewer_action("scroll_down")

    def action_logs_page_up(self) -> None:
        self._logs_viewer_action("page_up")

    def action_logs_page_down(self) -> None:
        self._logs_viewer_action("page_down")

    def _jump_tab(self, index: int) -> None:
        self._tabbed_content().active = self.TAB_ORDER[index]

    def _jump_tab_id(self, tab_id: str) -> None:
        self._jump_tab(self.TAB_ORDER.index(tab_id))

    def action_jump_tab_1(self) -> None:
        self._jump_tab(0)

    def action_jump_tab_2(self) -> None:
        self._jump_tab(1)

    def action_jump_tab_3(self) -> None:
        self._jump_tab(2)

    def action_jump_tab_4(self) -> None:
        self._jump_tab(3)

    def action_jump_tab_5(self) -> None:
        self._jump_tab(4)

    def action_jump_tab_6(self) -> None:
        self._jump_tab(5)

    def _active_tab_id(self) -> str | None:
        return self._tabbed_content().active

    def _eval_pane(self) -> EvalPane:
        return self.query_one(EvalPane)

    def _training_pane(self) -> TrainingPane:
        return self.query_one(TrainingPane)

    def _benchmarks_pane(self) -> BenchmarksPane:
        return self.query_one(BenchmarksPane)

    def _run_pane(self) -> RunPane:
        return self.query_one(RunPane)

    def _logs_pane(self) -> LogsPane:
        return self.query_one(LogsPane)

    def _settings_pane(self) -> SettingsPane:
        return self.query_one(SettingsPane)

    def action_refresh(self) -> None:
        self._training_pane().reload_data()
        self._eval_pane().reload_data()
        self._benchmarks_pane().reload_data()
        self._logs_pane().reload_data()
        self._logs_pane().force_refresh_opened_log()

    def action_filter(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-benchmarks":
            self._benchmarks_pane().action_open_filter()
        elif tab == "tab-run":
            self._run_pane().action_open_filter()
        elif tab == "tab-logs":
            self._logs_pane().action_open_filter()

    def action_clear_filter(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-settings":
            field = self._settings_pane().query_one("#settings-edit-input", Input)
            if field.has_class("visible"):
                field.value = ""
                field.remove_class("visible")
                self._settings_pane().query_one("#settings-table", SettingsDataTable).focus()
                return
        if tab == "tab-benchmarks":
            self._benchmarks_pane().action_clear_filter()
        elif tab == "tab-run":
            self._run_pane().action_clear_filter()
        elif tab == "tab-logs":
            logs = self._logs_pane()
            if logs.is_viewing_log():
                logs.action_back_to_list()
            else:
                logs.action_clear_filter()

    def action_bookmark(self) -> None:
        if self._active_tab_id() == "tab-logs":
            self._logs_pane().action_bookmark_active()

    def action_prev_item(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-logs" and self._logs_pane().is_viewing_log():
            return
        if tab == "tab-training":
            self._training_pane().action_prev_run()
        elif tab == "tab-eval":
            self._eval_pane().action_prev_run()
        elif tab == "tab-run":
            self._run_pane().action_prev_recipe()
        elif tab == "tab-settings":
            self._settings_pane().action_adjust_setting(-1)
        elif tab == "tab-logs":
            self._logs_pane().action_prev_file()

    def action_next_item(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-logs" and self._logs_pane().is_viewing_log():
            return
        if tab == "tab-training":
            self._training_pane().action_next_run()
        elif tab == "tab-eval":
            self._eval_pane().action_next_run()
        elif tab == "tab-run":
            self._run_pane().action_next_recipe()
        elif tab == "tab-settings":
            self._settings_pane().action_adjust_setting(1)
        elif tab == "tab-logs":
            self._logs_pane().action_next_file()

    def action_open_detail(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-settings":
            self._settings_pane().action_open_editor()
            return
        if tab == "tab-run":
            self.run_worker(self._run_pane().action_confirm_launch())
            return
        if tab != "tab-benchmarks":
            return
        bench = self._benchmarks_pane()
        run_id = bench.highlighted_run_id()
        if run_id is None:
            return
        self._eval_pane().open_run(bench.active_suite_name(), run_id)
        self._jump_tab_id("tab-eval")

    def action_launch(self) -> None:
        if self._active_tab_id() != "tab-run":
            raise SkipAction()
        self.run_worker(self._run_pane().action_confirm_launch())

    def action_open_last_launch_log(self) -> None:
        log_path = self._run_pane().last_launch_log()
        if log_path is None:
            self.notify("No launch in this session yet", title="Run", timeout=3)
            return
        self.follow_launch_log(log_path, LAUNCH_FOLLOW_LOGS)

    def follow_launch_log(self, log_path: Path, mode: str) -> None:
        """Open or tail the launch log according to the selected follow mode."""
        if mode == LAUNCH_FOLLOW_NONE:
            return
        if mode == LAUNCH_FOLLOW_TERMINAL:
            if tail_log_in_terminal(log_path, tail_lines=self.settings.tail_lines):
                self.notify(
                    f"tail -f in tmux split → {log_path.name}",
                    title="Following in terminal",
                    timeout=4,
                )
                return
            fallback = tail_log_command(log_path, tail_lines=self.settings.tail_lines)
            self.notify(
                f"Not in tmux — following in Logs tab. Manual: {fallback}",
                title="Following in Logs",
                timeout=8,
            )
            mode = LAUNCH_FOLLOW_LOGS
        if mode == LAUNCH_FOLLOW_LOGS:
            logs = self._logs_pane()
            logs.reload_data()
            logs._open_file(log_path)
            self._jump_tab_id("tab-logs")

    def action_increase_setting(self) -> None:
        if self._active_tab_id() == "tab-settings":
            self._settings_pane().action_adjust_setting(1)

    def action_decrease_setting(self) -> None:
        if self._active_tab_id() == "tab-settings":
            self._settings_pane().action_adjust_setting(-1)

    def action_toggle_setting(self) -> None:
        if self._active_tab_id() == "tab-settings":
            self._settings_pane().action_toggle_color()

    def action_restart(self) -> None:
        self.exit(RESTART_EXIT_CODE)

    def action_help(self) -> None:
        self.notify(
            "←→ Tab: tabs (not while viewing a log)  1-6: jump  ↑↓: row  [ ]: prev/next"
            "  Enter/x: launch (Run) / open log (Logs) / edit (Settings) / Benchmarks→Eval"
            "  f: follow mode (launch confirm)"
            "  + / -: adjust setting (Settings)  space: toggle  /: filter"
            "  Esc: back to list / clear filter  b: toggle bookmark (Logs)"
            "  r: refresh  Ctrl+R: restart  q: quit",
            title="rl-dbs-tui",
            timeout=6,
        )
