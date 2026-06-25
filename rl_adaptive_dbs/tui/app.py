"""``rl-dbs-tui`` Textual application."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane

from rl_adaptive_dbs.tui.data import (
    placeholder_tab_message,
    refresh_suites,
    select_suite,
    suite_status_line,
    suite_table_rows,
)


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
    """Read-only monitor — Benchmarks tab (Phase 4)."""

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
        Binding("question_mark", "help", "Help", show=False),
    ]

    def __init__(self, results_dir: Path, *, refresh_s: float = 1.0) -> None:
        super().__init__()
        self.results_dir = results_dir
        self.refresh_s = refresh_s

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Training", id="tab-training"):
                yield Static(placeholder_tab_message("Training"), classes="placeholder")
            with TabPane("Eval", id="tab-eval"):
                yield Static(placeholder_tab_message("Eval"), classes="placeholder")
            with TabPane("Benchmarks", id="tab-benchmarks"):
                yield BenchmarksPane(self.results_dir, refresh_s=self.refresh_s)
            with TabPane("Logs", id="tab-logs"):
                yield Static(placeholder_tab_message("Logs"), classes="placeholder")
        yield Footer()

    def action_refresh(self) -> None:
        pane = self.query_one(BenchmarksPane)
        pane.reload_data()

    def action_filter(self) -> None:
        self.query_one(BenchmarksPane).action_open_filter()

    def action_clear_filter(self) -> None:
        self.query_one(BenchmarksPane).action_clear_filter()

    def action_help(self) -> None:
        self.notify(
            "Tab: switch tabs  /: filter  r: refresh  q: quit",
            title="rl-dbs-tui",
            timeout=4,
        )
