"""``rl-dbs-tui`` Textual application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Input, ProgressBar, Sparkline, Static, TabbedContent, TabPane

from rl_adaptive_dbs.tui.data import (
    placeholder_tab_message,
    refresh_suites,
    select_suite,
    suite_status_line,
    suite_table_rows,
)
from rl_adaptive_dbs.tui.training_data import (
    cycle_training_run_id,
    discover_training_runs,
    return_sparkline_data,
    run_selector_rows,
    select_training_run,
    training_empty_message,
    training_metadata_lines,
    training_status_line,
)


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

    def __init__(self, artifacts_dir: Path, *, refresh_s: float = 1.0) -> None:
        super().__init__()
        self.artifacts_dir = artifacts_dir
        self.refresh_s = refresh_s
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
        table.add_columns("run", "seed", "episodes", "last_return")
        table.cursor_type = "row"
        self.set_interval(self.refresh_s, self.reload_data)
        self.reload_data()

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
            return

        empty.display = False
        table.display = True
        progress.display = True
        sparkline.display = True

        if run is None:
            status.update(f"No training logs under {self.artifacts_dir}/")
            return

        self._active_run_id = run.run_id
        status.update(training_status_line(run, self._runs))
        detail.update(
            f"episode return (last {min(len(run.episodes), 40)} episodes)"
            if run.episodes
            else "episode return"
        )

        table.clear()
        active_row: int | None = None
        for index, run_row in enumerate(self._runs):
            label, seed, episodes, last = run_selector_rows([run_row])[0]
            table.add_row(label, seed, episodes, last, key=run_row.run_id)
            if run_row.run_id == run.run_id:
                active_row = index
        if active_row is not None:
            table.move_cursor(row=active_row)

        progress.update(total=run.planned_episodes, progress=run.current_episode)
        sparkline.data = return_sparkline_data(run)
        meta.update("\n".join(training_metadata_lines(run)))

    def action_prev_run(self) -> None:
        self._active_run_id = cycle_training_run_id(self._runs, self._active_run_id, -1)
        self.reload_data()

    def action_next_run(self) -> None:
        self._active_run_id = cycle_training_run_id(self._runs, self._active_run_id, 1)
        self.reload_data()

    def _select_run_from_table(self, row_key) -> None:
        if row_key is None:
            return
        run_id = str(row_key.value)
        if run_id != self._active_run_id:
            self._active_run_id = run_id
            self.reload_data()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "training-run-table":
            return
        self._select_run_from_table(event.row_key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "training-run-table":
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
        self.set_interval(self.refresh_s, self.reload_data)
        self.reload_data()

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
        for row in suite_table_rows(suite, query=self._filter):
            table.add_row(*row)

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


class RlDbsTuiApp(App):
    """Read-only monitor — Training + Benchmarks tabs."""

    TITLE = "rl-dbs-tui"
    CSS = """
    TabbedContent {
        height: 1fr;
    }
    .placeholder {
        padding: 1 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("/", "filter", "Filter", show=True),
        Binding("escape", "clear_filter", "Clear filter"),
        Binding("[", "prev_item", "Prev", show=False),
        Binding("]", "next_item", "Next", show=False),
        Binding("question_mark", "help", "Help", show=False),
    ]

    def __init__(
        self,
        results_dir: Path,
        *,
        artifacts_dir: Path,
        refresh_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.artifacts_dir = artifacts_dir
        self.refresh_s = refresh_s

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Training", id="tab-training"):
                yield TrainingPane(self.artifacts_dir, refresh_s=self.refresh_s)
            with TabPane("Eval", id="tab-eval"):
                yield Static(placeholder_tab_message("Eval"), classes="placeholder")
            with TabPane("Benchmarks", id="tab-benchmarks"):
                yield BenchmarksPane(self.results_dir, refresh_s=self.refresh_s)
            with TabPane("Logs", id="tab-logs"):
                yield Static(placeholder_tab_message("Logs"), classes="placeholder")
        yield Footer()

    def _active_tab_id(self) -> str | None:
        tabs = self.query_one(TabbedContent)
        return tabs.active

    def _training_pane(self) -> TrainingPane:
        return self.query_one(TrainingPane)

    def _benchmarks_pane(self) -> BenchmarksPane:
        return self.query_one(BenchmarksPane)

    def action_refresh(self) -> None:
        self._training_pane().reload_data()
        self._benchmarks_pane().reload_data()

    def action_filter(self) -> None:
        if self._active_tab_id() == "tab-benchmarks":
            self._benchmarks_pane().action_open_filter()

    def action_clear_filter(self) -> None:
        self._benchmarks_pane().action_clear_filter()

    def action_prev_item(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-training":
            self._training_pane().action_prev_run()

    def action_next_item(self) -> None:
        tab = self._active_tab_id()
        if tab == "tab-training":
            self._training_pane().action_next_run()

    def action_help(self) -> None:
        self.notify(
            "Tab: tabs  [/]: run (Training)  /: filter (Benchmarks)  r: refresh  q: quit",
            title="rl-dbs-tui",
            timeout=5,
        )
