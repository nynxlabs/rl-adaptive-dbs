"""Training artifacts loader for the TUI Training tab."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_TOTAL_EPISODES = 10
MAX_EPISODE_POINTS = 10_000
MAX_SPARKLINE_EPISODES = 40

_EPISODE_LINE = re.compile(
    r"episode\s+(\d+)\s*/\s*(\d+)(?:,\s*reward:\s*([-\d.]+)|\s+reward=([-\d.]+))?",
    re.IGNORECASE,
)
_TRAIN_SEED = re.compile(r"(.+)_train(\d+)$")
_TRAIN_LOG_STEM = re.compile(r"train_(.+)_seed(\d+)$", re.IGNORECASE)
_LOG_SUFFIX_PRIORITY = {".jsonl": 0, ".json": 1, ".log": 2}


@dataclass(frozen=True)
class TrainEpisode:
    episode: int
    return_value: float
    loss: float | None = None
    timestamp: float | None = None


@dataclass
class TrainingRun:
    run_id: str
    controller: str
    variant: str
    seed: int | None
    log_path: Path
    episodes: list[TrainEpisode] = field(default_factory=list)
    total_episodes: int | None = None
    error: str | None = None
    mtime: float = 0.0

    @property
    def current_episode(self) -> int:
        if not self.episodes:
            return 0
        return max(entry.episode for entry in self.episodes)

    @property
    def planned_episodes(self) -> int:
        if self.total_episodes is not None:
            return self.total_episodes
        return max(DEFAULT_TOTAL_EPISODES, self.current_episode)

    @property
    def last_return(self) -> float | None:
        if not self.episodes:
            return None
        return self.episodes[-1].return_value

    @property
    def last_timestamp(self) -> float | None:
        if not self.episodes:
            return None
        return self.episodes[-1].timestamp


def discover_training_runs(artifacts_dir: Path) -> list[TrainingRun]:
    """Find training logs under ``artifacts_dir``, newest first."""
    if not artifacts_dir.is_dir():
        return []

    candidates: list[Path] = []
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".jsonl", ".json", ".log"}:
            continue
        if path.name == "train_log.jsonl" or _looks_like_training_log(path):
            candidates.append(path)

    runs: list[TrainingRun] = []
    for path in _dedupe_log_paths(candidates):
        run = load_training_run(path, artifacts_dir)
        if run is not None:
            runs.append(run)

    runs.sort(key=lambda run: run.mtime, reverse=True)
    return runs


def _dedupe_log_paths(paths: list[Path]) -> list[Path]:
    """Prefer JSONL/JSON over companion ``.log`` files for the same run stem."""
    best: dict[str, Path] = {}
    for path in paths:
        if path.name == "train_log.jsonl":
            key = path.parent.as_posix()
        else:
            key = path.with_suffix("").as_posix()
        existing = best.get(key)
        if existing is None or _log_path_priority(path) < _log_path_priority(existing):
            best[key] = path
    return list(best.values())


def _log_path_priority(path: Path) -> int:
    if path.name == "train_log.jsonl":
        return -1
    return _LOG_SUFFIX_PRIORITY.get(path.suffix.lower(), 9)


def _looks_like_training_log(path: Path) -> bool:
    name = path.name.lower()
    if "train" not in name:
        return False
    if path.suffix == ".log":
        return True
    if path.suffix == ".jsonl":
        return True
    if path.suffix != ".json":
        return False
    if name.endswith("_final.json"):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:4096]
    except OSError:
        return False
    text = text.lstrip()
    if not text.startswith("["):
        return False
    return '"episode"' in text or "'episode'" in text


def load_training_run(log_path: Path, artifacts_dir: Path) -> TrainingRun | None:
    controller, variant, seed = _infer_run_metadata(log_path, artifacts_dir)
    run_id = log_path.relative_to(artifacts_dir).as_posix()
    try:
        mtime = log_path.stat().st_mtime
    except OSError:
        mtime = 0.0

    episodes, total, error = parse_training_log(log_path)
    if error and not episodes:
        return TrainingRun(
            run_id=run_id,
            controller=controller,
            variant=variant,
            seed=seed,
            log_path=log_path,
            episodes=[],
            total_episodes=total,
            error=error,
            mtime=mtime,
        )
    if not episodes:
        return None

    if seed is None:
        seed = _seed_from_checkpoint(log_path.parent, variant)

    return TrainingRun(
        run_id=run_id,
        controller=controller,
        variant=variant,
        seed=seed,
        log_path=log_path,
        episodes=episodes,
        total_episodes=total,
        error=error,
        mtime=mtime,
    )


def parse_training_log(path: Path) -> tuple[list[TrainEpisode], int | None, str | None]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl" or path.name == "train_log.jsonl":
            return _parse_jsonl(path)
        if suffix == ".json":
            return _parse_json_array(path)
        if suffix == ".log":
            return _parse_text_log(path)
    except OSError as exc:
        return [], None, str(exc)
    return [], None, f"unsupported log format: {path.name}"


def _parse_jsonl(path: Path) -> tuple[list[TrainEpisode], int | None, str | None]:
    episodes: list[TrainEpisode] = []
    total: int | None = None
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return episodes, total, f"line {line_no}: {exc.msg}"
        if not isinstance(payload, dict):
            continue
        total = _coalesce_total(total, payload)
        episode = _episode_from_payload(payload)
        if episode is not None:
            episodes.append(episode)
    episodes = _trim_episodes(episodes)
    return episodes, total, None


def _parse_json_array(path: Path) -> tuple[list[TrainEpisode], int | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], None, exc.msg
    if not isinstance(payload, list):
        return [], None, "expected JSON array"
    episodes: list[TrainEpisode] = []
    total: int | None = None
    for item in payload:
        if not isinstance(item, dict):
            continue
        total = _coalesce_total(total, item)
        episode = _episode_from_payload(item)
        if episode is not None:
            episodes.append(episode)
    episodes = _trim_episodes(episodes)
    return episodes, total, None


def _parse_text_log(path: Path) -> tuple[list[TrainEpisode], int | None, str | None]:
    episodes: list[TrainEpisode] = []
    total: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _EPISODE_LINE.search(line)
        if not match:
            continue
        current = int(match.group(1))
        planned = int(match.group(2))
        total = planned
        reward_raw = match.group(3) or match.group(4)
        return_value = float(reward_raw) if reward_raw is not None else 0.0
        episodes.append(TrainEpisode(episode=current, return_value=return_value))
    episodes = _trim_episodes(episodes)
    return episodes, total, None


def _episode_from_payload(payload: dict[str, object]) -> TrainEpisode | None:
    raw_episode = payload.get("episode")
    if raw_episode is None:
        return None
    try:
        episode = int(raw_episode)
    except (TypeError, ValueError):
        return None

    return_value = payload.get("return", payload.get("reward"))
    if return_value is None:
        return None
    try:
        reward = float(return_value)
    except (TypeError, ValueError):
        return None

    loss = payload.get("loss")
    loss_value = float(loss) if loss is not None else None
    timestamp = payload.get("timestamp")
    ts_value = float(timestamp) if timestamp is not None else None
    return TrainEpisode(
        episode=episode,
        return_value=reward,
        loss=loss_value,
        timestamp=ts_value,
    )


def _coalesce_total(current: int | None, payload: dict[str, object]) -> int | None:
    for key in ("total_episodes", "episodes_total", "num_episodes"):
        value = payload.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return current


def _trim_episodes(episodes: list[TrainEpisode]) -> list[TrainEpisode]:
    if len(episodes) <= MAX_EPISODE_POINTS:
        return episodes
    return episodes[-MAX_EPISODE_POINTS:]


def _infer_run_metadata(log_path: Path, artifacts_dir: Path) -> tuple[str, str, int | None]:
    rel = log_path.relative_to(artifacts_dir)
    parts = rel.parts
    controller = parts[0] if len(parts) >= 2 else "unknown"
    variant = "unknown"
    seed: int | None = None

    if log_path.name == "train_log.jsonl" and len(parts) >= 3:
        return parts[0], parts[1], seed

    stem = log_path.stem
    match = _TRAIN_LOG_STEM.match(stem)
    if match:
        return controller, match.group(1), int(match.group(2))

    ckpt_match = _TRAIN_SEED.match(stem)
    if ckpt_match:
        variant = ckpt_match.group(1)
        seed = int(ckpt_match.group(2))
        return controller, variant, seed

    variant = stem.removesuffix("_final")
    return controller, variant, seed


def _seed_from_checkpoint(directory: Path, variant: str) -> int | None:
    for path in sorted(directory.glob(f"{variant}_train*.pt")):
        match = _TRAIN_SEED.match(path.stem)
        if match:
            return int(match.group(2))
    return None


def select_training_run(runs: list[TrainingRun], run_id: str | None) -> TrainingRun | None:
    if not runs:
        return None
    if run_id:
        for run in runs:
            if run.run_id == run_id:
                return run
    return runs[0]


def cycle_training_run_id(
    runs: list[TrainingRun],
    current: str | None,
    delta: int,
) -> str | None:
    if not runs:
        return None
    ids = [run.run_id for run in runs]
    if current not in ids:
        return ids[0]
    index = (ids.index(current) + delta) % len(ids)
    return ids[index]


def return_sparkline_data(run: TrainingRun, *, max_points: int = MAX_SPARKLINE_EPISODES) -> list[float]:
    values = [entry.return_value for entry in run.episodes]
    if len(values) <= max_points:
        return values
    return values[-max_points:]


def training_status_line(run: TrainingRun, runs: list[TrainingRun] | None = None) -> str:
    picker = ""
    if runs and len(runs) > 1 and run.run_id in [item.run_id for item in runs]:
        index = [item.run_id for item in runs].index(run.run_id) + 1
        picker = f"  [{index}/{len(runs)}]"
    switch = "  [/] run" if runs and len(runs) > 1 else ""
    seed = str(run.seed) if run.seed is not None else "?"
    return (
        f"Training: {run.controller}/{run.variant}{picker}{switch}  "
        f"seed: {seed}  episode: {run.current_episode}/{run.planned_episodes}"
    )


def training_metadata_lines(run: TrainingRun) -> list[str]:
    lines = [
        f"log: {run.log_path.name}",
    ]
    if run.error:
        lines.append(f"warning: {run.error}")
    if run.last_return is not None:
        lines.append(f"last return: {run.last_return:.2f}")
    if run.last_timestamp is not None:
        lines.append(f"updated: {_format_timestamp(run.last_timestamp)}")
    return lines


def training_empty_message(artifacts_dir: Path) -> str:
    return (
        f"No training logs in {artifacts_dir}/.\n\n"
        "Start training with `uv run rl-dbs train ...` or a training script "
        "that writes `train_log.jsonl` (or episode JSON/logs) under artifacts/."
    )


def run_selector_rows(runs: list[TrainingRun]) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for run in runs:
        episodes = f"{run.current_episode}/{run.planned_episodes}"
        last = f"{run.last_return:.1f}" if run.last_return is not None else "n/a"
        label = f"{run.controller}/{run.variant}"
        if run.error and not run.episodes:
            last = "ERR"
        rows.append((label, str(run.seed or "?"), episodes, last))
    return rows


def _format_timestamp(value: float) -> str:
    if value > 1_000_000_000:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    return str(value)
