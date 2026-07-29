"""Refresh ``figures/mehregan/replications.md`` after paper figure plot scripts run.

Plot scripts write replication PNGs under ``figures/<paper>/images/`` and JSON caches under
``artifacts/figures/papers/``. This module updates caption markers + replication image
links in the comparison doc.

When run from a git worktree under ``.worktrees/``, shared doc/figure paths resolve to
the **main checkout** (vault symlink for ``figures/<paper>/replications.md``) so promote does not leave a
detached worktree copy of the index.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any


def main_checkout_root(checkout: Path | None = None) -> Path:
    """Return the main repo checkout if ``checkout`` is ``.../.worktrees/<name>``."""
    resolved = (checkout or Path(__file__).resolve().parents[3]).resolve()
    parts = resolved.parts
    if ".worktrees" in parts:
        idx = parts.index(".worktrees")
        if idx > 0:
            return Path(*parts[:idx])
    return resolved


CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = main_checkout_root(CHECKOUT_ROOT)


def resolve_paper_1_doc(checkout: Path | None = None) -> Path:
    """Path to ``figures/mehregan/replications.md`` that plot/promote should update.

    Prefers the main checkout path (usually a vault symlink) so worktree runs update
    the shared doc the main branch and Obsidian see.
    """
    root = main_checkout_root(checkout or CHECKOUT_ROOT)
    return root / "figures" / "mehregan" / "replications.md"


PAPER_1_DOC = resolve_paper_1_doc()


def resolve_nguyen_doc(checkout: Path | None = None) -> Path:
    """Path to ``figures/nguyen/replications.md`` (Nguyen tracker)."""
    root = main_checkout_root(checkout or CHECKOUT_ROOT)
    return root / "figures" / "nguyen" / "replications.md"


PAPER_NGUYEN_DOC = resolve_nguyen_doc()
# Back-compat alias used by promote_nguyen_2_3
PAPER_2_DOC = PAPER_NGUYEN_DOC
PAPER_NGUYEN_2_3_MANIFEST = "artifacts/figures/papers/nguyen/3/manifest.json"
PAPER_NGUYEN_2_3_REPLICATION_ALT = "Replication Fig 3"
PAPER_NGUYEN_4_MANIFEST = "artifacts/figures/papers/nguyen/4/manifest.json"
PAPER_NGUYEN_4_REPLICATION_ALT = "Replication Fig 4"

PAPER_1B_PNG = "figures/mehregan/images/1b/gpi_psd.png"
PAPER_1B_MANIFEST = "artifacts/figures/papers/mehregan/1b/manifest.json"
PAPER_2A_PNG = "figures/mehregan/images/2a/beta_power.png"
PAPER_2A_MANIFEST = "artifacts/figures/papers/mehregan/2a/manifest.json"
PAPER_2B_PNG = "figures/mehregan/images/2b/error_index_v1.png"
PAPER_2B_MANIFEST = "artifacts/figures/papers/mehregan/2b/manifest.json"
PAPER_4A_PNG = "figures/mehregan/images/4a/training_beta.png"
PAPER_4A_MANIFEST = "artifacts/figures/papers/mehregan/4a/manifest.json"
PAPER_4B_PNG = "figures/mehregan/images/4b/training_reward.png"
PAPER_4B_MANIFEST = "artifacts/figures/papers/mehregan/4b/manifest.json"
PAPER_1B_REF = "figures/mehregan/images/1b/paper.png"
PAPER_2A_REF = "figures/mehregan/images/2a/paper.png"
PAPER_2B_REF = "figures/mehregan/images/2b/paper.png"
PAPER_2B_REPLICATION_ALT = "Replication Fig 2b"
PAPER_4A_REF = "figures/mehregan/images/4a/paper.png"
PAPER_4A_REPLICATION_ALT = "Replication Fig 4a"
PAPER_4B_REF = "figures/mehregan/images/4b/paper.png"
PAPER_4B_REPLICATION_ALT = "Replication Fig 4b reward"
PAPER_4B_PSD_REPLICATION_ALT = "Replication Fig 4b PSD"
PAPER_5A_MANIFEST = "artifacts/figures/papers/mehregan/5a/manifest.json"
PAPER_5A_REPLICATION_ALT = "Replication Fig 5a"
PAPER_5B_MANIFEST = "artifacts/figures/papers/mehregan/5b/manifest.json"
PAPER_5B_REPLICATION_ALT = "Replication Fig 5b"
PAPER_6A_PNG = "figures/mehregan/images/6a/ptq_qat_45hz.png"
PAPER_6A_MANIFEST = "artifacts/figures/papers/mehregan/6a/manifest.json"
PAPER_6A_REF = "figures/mehregan/images/6a/paper.png"
PAPER_6A_REPLICATION_ALT = "Replication Fig 6a"
PAPER_6B_PNG = "figures/mehregan/images/6b/ptq_qat_30hz.png"
PAPER_6B_MANIFEST = "artifacts/figures/papers/mehregan/6b/manifest.json"
PAPER_6B_REF = "figures/mehregan/images/6b/paper.png"
PAPER_6B_REPLICATION_ALT = "Replication Fig 6b"

_VERSIONED_PNG_RE = re.compile(r"^(?P<stem>.+)_v(?P<ver>\d+)\.png$")


def latest_png_version(directory: Path, stem: str) -> int:
    """Highest ``{stem}_vN.png`` version in ``directory``, or 0 if none exist."""
    if not directory.is_dir():
        return 0
    best = 0
    for path in directory.iterdir():
        if not path.is_file() and not path.is_symlink():
            continue
        match = _VERSIONED_PNG_RE.match(path.name)
        if match is None or match.group("stem") != stem:
            continue
        best = max(best, int(match.group("ver")))
    return best


def next_versioned_png(directory: Path, stem: str) -> tuple[Path, int]:
    """Allocate the next ``{stem}_vN.png`` path (N = max existing + 1)."""
    version = latest_png_version(directory, stem) + 1
    return directory / f"{stem}_v{version}.png", version


def parse_png_version(path: Path) -> int | None:
    match = _VERSIONED_PNG_RE.match(path.name)
    return int(match.group("ver")) if match else None


def repo_rel_posix(path: Path) -> str:
    """Normalize a figure path to repo-relative POSIX (``figures/...``)."""
    text = path.as_posix()
    marker = "figures/"
    idx = text.find(marker)
    if idx >= 0:
        return text[idx:]
    return text


def _doc_figure_link(repo_rel: str) -> str:
    """Markdown image URL from a replication tracker to a repo ``figures/...`` asset."""
    if repo_rel.startswith("figures/"):
        tail = repo_rel.removeprefix("figures/")
        paper, _, rest = tail.partition("/")
        if rest.startswith("images/"):
            return rest
    return repo_rel


# Mirror PNGs into the main checkout (vault-backed) even when plotting from a worktree.
DOCS_FIGURE_PAPERS = REPO_ROOT / "docs" / "figures" / "papers"
CANONICAL_FIGURE_PAPERS = REPO_ROOT / "figures" / "papers"


def materialize_docs_figure_papers() -> int:
    """No-op: replication PNGs live under ``figures/<paper>/images/`` only."""
    return 0


def _set_markdown_image_link(text: str, *, alt: str, repo_rel: str) -> str:
    """Replace ``![alt](...)`` with a docs/figures-local link (``papers/...``)."""
    link = _doc_figure_link(repo_rel)
    pattern = re.compile(rf"!\[{re.escape(alt)}\]\([^)]+\)")
    if not pattern.search(text):
        msg = f"missing markdown image for alt={alt!r} in {PAPER_1_DOC}"
        raise ValueError(msg)
    return pattern.sub(f"![{alt}]({link})", text, count=1)


def _today() -> str:
    return date.today().isoformat()


def _caption_1b(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    if not seeds:
        seed_note = "see manifest"
    elif len(seeds) == 1:
        seed_note = f"seed {seeds[0]}"
    else:
        seed_note = f"seeds {seeds[0]}–{seeds[-1]} mean"
    duration = manifest.get("duration_s", 10)
    return f"{seed_note}, {duration:.0f} s segment ({_today()})"


def _caption_2a(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    seed_note = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {','.join(map(str, seeds))}"
    sampling = manifest.get("sampling", "trailing")
    if sampling == "segment":
        seg = manifest.get("segment_s", 2)
        protocol = f"{seg:.0f} s whole-segment bins"
    else:
        step = manifest.get("step_s", 0.2)
        window = manifest.get("window_s", 2.0)
        warmup = manifest.get("warmup_s")
        if warmup:
            protocol = (
                f"14 s sim (2 s pre-roll), plot = sim − 2 s, "
                f"{step:g} s trailing / {window:g} s window (end sim 14 s)"
            )
        else:
            protocol = f"{step:g} s trailing / {window:g} s window"
    return f"{protocol}, {seed_note} ({_today()})"


def papers_tracker_image_link(png_path: Path, *, doc: Path | None = None) -> str:
    """Markdown image path relative to a paper ``replications.md`` tracker."""
    import os

    repo_rel = repo_rel_posix(Path(png_path))
    tracker = (doc or PAPER_1_DOC).resolve()
    abs_png = (REPO_ROOT / repo_rel).resolve()
    return Path(os.path.relpath(abs_png, tracker.parent)).as_posix()


def _set_papers_tracker_image_link(
    text: str,
    *,
    alt: str,
    link: str,
    doc: Path,
) -> str:
    pattern = re.compile(rf"!\[{re.escape(alt)}\]\([^)]+\)")
    if not pattern.search(text):
        msg = f"missing markdown image for alt={alt!r} in {doc}"
        raise ValueError(msg)
    return pattern.sub(f"![{alt}]({link})", text, count=1)


def _replace_marker_in(text: str, marker: str, body: str, *, doc: Path) -> str:
    pattern = re.compile(
        rf"(<!-- {marker}:start -->)(.*?)(<!-- {marker}:end -->)",
        re.DOTALL,
    )
    if not pattern.search(text):
        msg = f"missing marker block {marker} in {doc}"
        raise ValueError(msg)
    return pattern.sub(rf"\1\n{body}\n\3", text, count=1)


def _replace_marker(text: str, marker: str, body: str) -> str:
    return _replace_marker_in(text, marker, body, doc=PAPER_1_DOC)


def _default_paper_1_doc(*, caption_1b: str, caption_2a: str) -> str:
    link_1b_ref = _doc_figure_link(PAPER_1B_REF)
    link_1b_png = _doc_figure_link(PAPER_1B_PNG)
    link_2a_ref = _doc_figure_link(PAPER_2A_REF)
    link_2a_png = _doc_figure_link(PAPER_2A_PNG)
    return f"""# Mehregan et al.  — figure comparisons

Side-by-side **paper panel** vs **our replication** for qualitative checks. Plot scripts write replication PNGs to `figures/mehregan/images/`; JSON caches to `artifacts/figures/papers/`.

| Panel | Script | Spec |
|-------|--------|------|
| Fig 1b — GPi PSD | `scripts/figures/papers/mehregan/1b/plot.py` | [plant.md](../plant.md) |
| Fig 2a — GPi $P_\\beta$ time series | `scripts/figures/papers/mehregan/2a/plot.py` | [plant.md](../plant.md) |

---

## Fig 1b — GPi PSD

Mean GPi multitaper power spectral density (1–50 Hz) for three conditions: **healthy control**, **PD no treatment**, and **PD + 130 Hz STN cDBS**. In the paper panel, untreated PD shows a sharp beta-band peak (~12–13 Hz) well above healthy; 130 Hz cDBS suppresses that peak while leaving a smaller secondary bump at higher frequency. Our replication should preserve the ordering **PD > healthy** on beta power and **130 Hz cDBS < PD** (see [plant.md](../plant.md); seeds 0–9 mean in manifest).

### Paper (Mehregan et al.)

![Paper Fig 1b]({link_1b_ref})

### Replication

![Replication Fig 1b]({link_1b_png})

<!-- caption-1b:start -->
**Caption:** {caption_1b}

**Manifest:** `{PAPER_1B_MANIFEST}`
<!-- caption-1b:end -->

---

## Fig 2a — GPi $P_\\beta$ time series

GPi beta-band power ($P_\\beta$, Eq. 1, 13–35 Hz) over **12 s**: **PD no treatment** (red) vs **PD + 130 Hz cDBS** (blue). Both traces share a high, overlapping baseline for **0–2 s** (no STN stimulation). A dashed vertical at **2 s** marks cDBS onset for the blue trace only. After onset, blue falls quickly to a low plateau; red stays elevated. Dense continuous lines (not 2 s step bins).

### Paper (Mehregan et al.)

![Paper Fig 2a]({link_2a_ref})

### Replication

![Replication Fig 2a]({link_2a_png})

<!-- caption-2a:start -->
**Caption:** {caption_2a}

**Manifest:** `{PAPER_2A_MANIFEST}`
<!-- caption-2a:end -->

### Side-by-side checklist

Qualitative gates first; numeric bands are approximate (paper read from panel; replication from `{PAPER_2A_MANIFEST}`, seed 0).

| Check | Paper | Replication | Match? |
|-------|-------|-------------|--------|
| **Plot style** | Dense line trace, 0–12 s | Dense trailing line (`--sampling trailing`, 61 points) | ✓ |
| **Axes** | Time (sec) 0–12; y **PSD** ~100–600 | Same labels and range | ✓ |
| **Legend** | PD no Treatment; PD 130 Hz Treatment | Same labels and colors | ✓ |
| **DBS onset marker** | Dashed vertical at **2 s** | Dashed vertical at 2 s | ✓ |
| **0–2 s overlap** | Red and blue identical | Identical (max Δ = 0) | ✓ |
| **$t=0$ baseline** | ~460–470 (high, not zero) | ~513 | ✓ (same band) |
| **$t=2$ level** | ~480 | ~472 | ✓ |
| **Blue drop after 2 s** | Sharp fall; ~300 by $t \\approx 3$ | ~298 at $t=3$ | ✓ |
| **Blue floor** | ~160–170 by $t \\approx 4$; ripple ~160–210 | ~150 at $t=4$; ~160–170 for $t=4$–12 | ✓ |
| **Red after 2 s** | Stays high ~430–500; no downward trend | High ~440–530; wiggles through $t=12$ | ✓ |
| **Separation after onset** | Blue clearly below red | Large gap from $t \\approx 3$ onward | ✓ |
| **$t=12$ red** | ~435 (still high) | **~503** (window sim `[12, 14]`) | ~✓ (slightly high) |
| **$t=12$ blue** | ~185 (stable low floor) | **~160** (ripple, not flat) | ~✓ (still ~25 below paper) |
| **End behavior** | Both traces wiggle at $t=12$ | Sliding windows through sim 14; no flat tail | ✓ |

**Protocol (2026-07-11):** trailing windows end at sim **14 s**; Fig 2a alone passes enlarged GPI spike buffer (904). Re-run: `uv run python scripts/figures/papers/mehregan/2a/plot.py`.

**Remaining gaps:** blue floor ~25 below paper at $t=12$; single seed (0).
"""


def _caption_2b(manifest: dict[str, Any]) -> str:
    seeds = manifest.get("seeds", [])
    seed_note = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {','.join(map(str, seeds))}"
    step = manifest.get("step_s", 0.2)
    window = manifest.get("window_s", 2.0)
    smc_schedule = manifest.get("smc_schedule", "periodic")
    smc_hz = manifest.get("smc_frequency_hz", 10.0)
    backend = manifest.get("plant_backend", "python")
    smc_site = manifest.get("smc_site", "thalamic")
    smc_pulse_source = manifest.get("smc_pulse_source", "drive")
    if smc_schedule == "boc":
        site = "Iappco" if smc_site == "cortical" else "Iappth"
        smc_note = f"SMC BoC inv-gamma {site}"
        if smc_pulse_source == "cor_spikes":
            smc_note += ", Cor-spike SMCτ"
    elif smc_schedule == "periodic":
        smc_note = f"SMC {smc_hz:g} Hz periodic"
    else:
        smc_note = "SMC off"
    version = manifest.get("png_version")
    protocol = (
        f"14 s sim (2 s pre-roll), plot = sim − 2 s, "
        f"{step:g} s trailing / {window:g} s EI window (end sim 14 s), "
        f"{smc_note}, backend {backend}"
    )
    bits = [protocol, seed_note]
    if version is not None:
        bits.append(f"v{version}")
    return f"{', '.join(bits)} ({_today()})"


def _caption_4a(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    mean_hz = manifest.get("mean_hz", 45)
    state_length = manifest.get("state_length", 1)
    state_mode = manifest.get("state_mode", "within_step")
    reward_state_mode = manifest.get("reward_state_mode", "full_segment")
    exploration = manifest.get("exploration_mode", "softmax")
    critic = manifest.get("critic_action_input", "one_hot")
    init_bias = manifest.get("init_bias_scale")
    version = manifest.get("png_version")
    summary = manifest.get("summary") or {}
    trend = summary.get("trend_down")
    early = summary.get("early_mean_0_130")
    late = summary.get("late_mean_150_end")
    bits = [
        f"{mean_hz:g} Hz fixed_mean_pattern",
        f"{state_mode} L={state_length}",
        f"reward={reward_state_mode}",
        exploration,
        f"critic={critic}",
        f"seed {seed}",
    ]
    if version is not None:
        bits.append(f"v{version}")
    if isinstance(init_bias, (int, float)):
        bits.append(f"init_bias={init_bias:g}")
    if isinstance(early, (int, float)) and isinstance(late, (int, float)):
        bits.append(f"early={early:.3f} late={late:.3f}")
    if trend is True:
        bits.append("trend↓")
    elif trend is False:
        bits.append("trend flat/↑")
    return f"{', '.join(bits)} ({_today()})"


def _caption_4b(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    fig4a_series = manifest.get("fig4a_series", "artifacts/figures/papers/mehregan/4a/series_v4.json")
    version = manifest.get("png_version")
    n_ep = manifest.get("num_episodes", 8)
    summary = manifest.get("summary") or {}
    fig4b_pass = summary.get("fig4b_pass") or {}
    ep1 = fig4b_pass.get("ep1")
    ep_last = fig4b_pass.get("ep_last")
    rise = summary.get("rise_episode")
    beta = manifest.get("episode_mean_beta") or []
    bits = [
        f"{n_ep} episodes",
        "45 Hz fixed_mean_pattern (Fig 4a paired run)",
        f"seed {seed}",
        f"source {Path(str(fig4a_series)).name}",
    ]
    if version is not None:
        bits.append(f"v{version}")
    if isinstance(ep1, (int, float)) and isinstance(ep_last, (int, float)):
        n0 = int(manifest.get("num_episodes", 8)) - 1
        bits.append(f"reward ep0={ep1:.1f} ep{n0}={ep_last:.1f}")
    if rise is not None:
        bits.append(f"rise_ep={rise}")
    if len(beta) >= 2:
        bits.append(f"psd {beta[0]:.3f}→{beta[-1]:.3f}")
    if summary.get("automation_pass"):
        bits.append("gate pass")
    return f"{', '.join(bits)} ({_today()})"


def _caption_block(caption: str, manifest_path: str) -> str:
    return f"**Caption:** {caption}\n\n**Manifest:** `{manifest_path}`"


def _read_caption_from_doc(marker: str) -> str | None:
    if not PAPER_1_DOC.exists():
        return None
    text = PAPER_1_DOC.read_text()
    pattern = re.compile(
        rf"<!-- {marker}:start -->\n\*\*Caption:\*\* (.*?)\n\n\*\*Manifest:",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _ensure_paper_1_doc(
    *,
    caption_1b: str | None = None,
    caption_2a: str | None = None,
    caption_2b: str | None = None,
    caption_4a: str | None = None,
) -> None:
    cap_1b = caption_1b or _read_caption_from_doc("caption-1b") or "see manifest"
    cap_2a = caption_2a or _read_caption_from_doc("caption-2a") or "see manifest"
    if not PAPER_1_DOC.exists():
        PAPER_1_DOC.parent.mkdir(parents=True, exist_ok=True)
        PAPER_1_DOC.write_text(_default_paper_1_doc(caption_1b=cap_1b, caption_2a=cap_2a))
        return
    text = PAPER_1_DOC.read_text()
    if caption_1b is not None:
        text = _replace_marker(
            text,
            "caption-1b",
            _caption_block(caption_1b, PAPER_1B_MANIFEST),
        )
    if caption_2a is not None:
        text = _replace_marker(
            text,
            "caption-2a",
            _caption_block(caption_2a, PAPER_2A_MANIFEST),
        )
    if caption_2b is not None and "<!-- caption-2b:start -->" in text:
        text = _replace_marker(
            text,
            "caption-2b",
            _caption_block(caption_2b, PAPER_2B_MANIFEST),
        )
    if caption_4a is not None and "<!-- caption-4a:start -->" in text:
        text = _replace_marker(
            text,
            "caption-4a",
            _caption_block(caption_4a, PAPER_4A_MANIFEST),
        )
    PAPER_1_DOC.write_text(text)


def _ensure_paper_1_doc_4b(*, caption_4b: str) -> None:
    if not PAPER_1_DOC.exists():
        return
    text = PAPER_1_DOC.read_text()
    if "<!-- caption-4b:start -->" not in text:
        return
    text = _replace_marker(
        text,
        "caption-4b",
        _caption_block(caption_4b, PAPER_4B_MANIFEST),
    )
    PAPER_1_DOC.write_text(text)


def promote_1b(
    *,
    manifest: dict[str, Any],
    curves_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 1b caption in ``figures/mehregan/replications.md``."""
    caption = _caption_1b(manifest)
    if update_docs:
        _ensure_paper_1_doc(caption_1b=caption, caption_2a=None, caption_2b=None)
    return {
        "png": str(png_path),
        "manifest": PAPER_1B_MANIFEST,
        "curves": str(curves_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_2a(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 2a caption in ``figures/mehregan/replications.md``."""
    caption = _caption_2a(manifest)
    if update_docs:
        _ensure_paper_1_doc(caption_1b=None, caption_2a=caption, caption_2b=None)
    return {
        "png": str(png_path),
        "manifest": str(PAPER_2A_MANIFEST),
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_2b(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 2b caption + replication image link in ``figures/mehregan/replications.md``."""
    caption = _caption_2b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _ensure_paper_1_doc(caption_1b=None, caption_2a=None, caption_2b=caption)
        text = PAPER_1_DOC.read_text()
        text = _set_markdown_image_link(
            text,
            alt=PAPER_2B_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_2B_MANIFEST,
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_4b(
    *,
    manifest: dict[str, Any],
    rewards_path: Path,
    reward_png_path: Path,
    psd_png_path: Path,
    update_docs: bool = True,
    png_path: Path | None = None,
) -> dict[str, str]:
    """Refresh Fig 4b caption + replication image links in ``figures/mehregan/replications.md``."""
    if png_path is not None:
        reward_png_path = png_path
    caption = _caption_4b(manifest)
    reward_rel = repo_rel_posix(reward_png_path)
    psd_rel = repo_rel_posix(psd_png_path)
    if update_docs:
        _ensure_paper_1_doc_4b(caption_4b=caption)
        text = PAPER_1_DOC.read_text()
        reward_link = _doc_figure_link(reward_rel)
        psd_link = _doc_figure_link(psd_rel)
        repl_block = (
            f"**Reward vs episode**\n\n"
            f"![Replication Fig 4b reward]({reward_link})\n\n"
            f"**Episode-mean PSD vs episode**\n\n"
            f"![Replication Fig 4b PSD]({psd_link})\n"
        )
        repl_start = text.find("### Replication\n\n", text.find("## Fig 4b"))
        if repl_start >= 0:
            repl_start += len("### Replication\n\n")
            caption_start = text.find("\n<!-- caption-4b:start -->", repl_start)
            if caption_start >= 0:
                text = text[:repl_start] + repl_block + text[caption_start:]
        text = _set_markdown_image_link(
            text,
            alt=PAPER_4B_REPLICATION_ALT,
            repo_rel=reward_rel,
        )
        if PAPER_4B_PSD_REPLICATION_ALT in text:
            text = _set_markdown_image_link(
                text,
                alt=PAPER_4B_PSD_REPLICATION_ALT,
                repo_rel=psd_rel,
            )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(reward_png_path),
        "reward_png_repo_rel": reward_rel,
        "psd_png_repo_rel": psd_rel,
        "manifest": PAPER_4B_MANIFEST,
        "rewards": str(rewards_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_4a(
    *,
    manifest: dict[str, Any],
    series_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 4a caption + replication image link in ``figures/mehregan/replications.md``."""
    caption = _caption_4a(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _ensure_paper_1_doc(
            caption_1b=None,
            caption_2a=None,
            caption_2b=None,
            caption_4a=caption,
        )
        text = PAPER_1_DOC.read_text()
        text = _set_markdown_image_link(
            text,
            alt=PAPER_4A_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_4A_MANIFEST,
        "series": str(series_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def _caption_5a(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = manifest.get("gates") or panel.get("gates") or {}
    ckpt = manifest.get("checkpoint") or manifest.get("fig4a_checkpoint") or "see manifest"
    bits = [
        "45 Hz paper-protocol eval",
        f"seed {seed}",
        f"checkpoint={Path(str(ckpt)).name}",
    ]
    if manifest.get("skip_regular"):
        bits.append("skip_regular")
    if manifest.get("sampling") == "trailing":
        bits.append("0.2s trailing")
    version = manifest.get("png_version")
    if version is not None:
        bits.append(f"v{version}")
    trained_mean = panel.get("trained_mean")
    if isinstance(trained_mean, (int, float)):
        bits.append(f"trained_mean={trained_mean:.0f}")
    bits.append(f"no_stim_mean={panel.get('no_stim_mean', float('nan')):.0f}")
    periodic_mean = panel.get("periodic_mean")
    if isinstance(periodic_mean, (int, float)):
        bits.append(f"periodic_mean={periodic_mean:.0f}")
    if gates.get("trained_above_periodic"):
        bits.append("trained>periodic")
    elif panel.get("trained_equals_periodic"):
        bits.append("trained≡periodic")
    if gates.get("pass"):
        bits.append("gates pass")
    else:
        bits.append("gates open")
    return f"{', '.join(bits)} ({_today()})"


def _ensure_paper_1_doc_5a(*, caption_5a: str) -> None:
    if not PAPER_1_DOC.exists():
        return
    text = PAPER_1_DOC.read_text()
    if "<!-- caption-5a:start -->" not in text:
        return
    text = _replace_marker(
        text,
        "caption-5a",
        _caption_block(caption_5a, PAPER_5A_MANIFEST),
    )
    PAPER_1_DOC.write_text(text)


def _caption_5b(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = manifest.get("gates") or panel.get("gates") or {}
    ckpt = manifest.get("checkpoint") or "see manifest"
    bits = [
        "30 Hz paper-protocol eval",
        f"seed {seed}",
        f"checkpoint={Path(str(ckpt)).name}",
    ]
    if manifest.get("sampling") == "trailing":
        bits.append("0.2s trailing")
    version = manifest.get("png_version")
    if version is not None:
        bits.append(f"v{version}")
    trained_mean = panel.get("trained_mean")
    if isinstance(trained_mean, (int, float)):
        bits.append(f"trained_mean={trained_mean:.0f}")
    bits.append(f"no_stim_mean={panel.get('no_stim_mean', float('nan')):.0f}")
    periodic_mean = panel.get("periodic_mean")
    if isinstance(periodic_mean, (int, float)):
        bits.append(f"periodic_mean={periodic_mean:.0f}")
    if gates.get("trained_below_periodic") and gates.get("trained_below_no_stim"):
        bits.append("trained<both")
    elif panel.get("trained_equals_periodic"):
        bits.append("trained≡periodic")
    if gates.get("pass"):
        bits.append("gates pass")
    else:
        bits.append("gates open")
    return f"{', '.join(bits)} ({_today()})"


def _ensure_paper_1_doc_5b(*, caption_5b: str) -> None:
    if not PAPER_1_DOC.exists():
        return
    text = PAPER_1_DOC.read_text()
    if "<!-- caption-5b:start -->" not in text:
        return
    text = _replace_marker(
        text,
        "caption-5b",
        _caption_block(caption_5b, PAPER_5B_MANIFEST),
    )
    PAPER_1_DOC.write_text(text)


def promote_5b(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 5b caption + replication image link in ``figures/mehregan/replications.md``."""
    caption = _caption_5b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _ensure_paper_1_doc_5b(caption_5b=caption)
        text = PAPER_1_DOC.read_text()
        text = _set_markdown_image_link(
            text,
            alt=PAPER_5B_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_5B_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_5a(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 5a caption + replication image link in ``figures/mehregan/replications.md``."""
    caption = _caption_5a(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _ensure_paper_1_doc_5a(caption_5a=caption)
        text = PAPER_1_DOC.read_text()
        repl_old = "*Not yet generated.* Target: `figures/mehregan/images/5a/efficacy_45hz.png`"
        if repl_old in text:
            text = text.replace(
                repl_old,
                f"![Replication Fig 5a]({_doc_figure_link(repo_rel)})",
                1,
            )
        text = _set_markdown_image_link(
            text,
            alt=PAPER_5A_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_5A_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def _caption_6a(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = panel.get("gates") or {}
    bits = [
        "45 Hz paper-protocol eval",
        f"seed {seed}",
        f"fp32_post={panel.get('fp32_post_mean', 0):.0f}",
        f"qat_post={panel.get('qat_post_mean', 0):.0f}",
    ]
    if gates.get("ptq-fp16_tracks_fp32") and gates.get("ptq-int8_tracks_fp32"):
        bits.append("PTQ tracks fp32")
    if gates.get("qat_elevated_vs_fp32"):
        bits.append("QAT elevated")
    bits.append(_today())
    return ", ".join(bits)


def _ensure_paper_1_doc_6a(*, caption_6a: str) -> None:
    if not PAPER_1_DOC.exists():
        return
    text = PAPER_1_DOC.read_text()
    if "<!-- caption-6a:start -->" not in text:
        return
    text = _replace_marker(
        text,
        "caption-6a",
        _caption_block(caption_6a, PAPER_6A_MANIFEST),
    )
    PAPER_1_DOC.write_text(text)


def promote_6a(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 6a caption + replication image link in ``figures/mehregan/replications.md``."""
    caption = _caption_6a(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _ensure_paper_1_doc_6a(caption_6a=caption)
        text = PAPER_1_DOC.read_text()
        repl_old = "*Not yet generated.* Target: `figures/mehregan/images/6a/ptq_qat_45hz.png`"
        if repl_old in text:
            text = text.replace(
                repl_old,
                f"![Replication Fig 6a]({_doc_figure_link(repo_rel)})",
                1,
            )
        text = _set_markdown_image_link(
            text,
            alt=PAPER_6A_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_6A_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def _caption_6b(manifest: dict[str, Any]) -> str:
    seed = manifest.get("seed", 0)
    panel = manifest.get("panel") or {}
    gates = panel.get("gates") or {}
    bits = [
        "30 Hz paper-protocol eval",
        f"seed {seed}",
        f"fp32_post={panel.get('fp32_post_mean', 0):.0f}",
        f"qat_post={panel.get('qat_post_mean', 0):.0f}",
    ]
    if gates.get("ptq-fp16_tracks_fp32") and gates.get("ptq-int8_tracks_fp32"):
        bits.append("PTQ tracks fp32")
    if gates.get("qat_elevated_vs_fp32"):
        bits.append("QAT elevated")
    bits.append(_today())
    return ", ".join(bits)


def _ensure_paper_1_doc_6b(*, caption_6b: str) -> None:
    if not PAPER_1_DOC.exists():
        return
    text = PAPER_1_DOC.read_text()
    if "<!-- caption-6b:start -->" not in text:
        return
    text = _replace_marker(
        text,
        "caption-6b",
        _caption_block(caption_6b, PAPER_6B_MANIFEST),
    )
    PAPER_1_DOC.write_text(text)


def promote_6b(
    *,
    manifest: dict[str, Any],
    eval_path: Path,
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 6b caption + replication image link in ``figures/mehregan/replications.md``."""
    caption = _caption_6b(manifest)
    repo_rel = repo_rel_posix(png_path)
    if update_docs:
        _ensure_paper_1_doc_6b(caption_6b=caption)
        text = PAPER_1_DOC.read_text()
        repl_old = "*Not yet generated.* Target: `figures/mehregan/images/6b/ptq_qat_30hz.png`"
        if repl_old in text:
            text = text.replace(
                repl_old,
                f"![Replication Fig 6b]({_doc_figure_link(repo_rel)})",
                1,
            )
        text = _set_markdown_image_link(
            text,
            alt=PAPER_6B_REPLICATION_ALT,
            repo_rel=repo_rel,
        )
        PAPER_1_DOC.write_text(text)
    materialize_docs_figure_papers()
    return {
        "png": str(png_path),
        "png_repo_rel": repo_rel,
        "manifest": PAPER_6B_MANIFEST,
        "eval": str(eval_path),
        "caption": caption,
        "doc": str(PAPER_1_DOC),
    }


def promote_nguyen_2_3(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 3 caption + replication image link in ``figures/nguyen/replications.md``."""
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    link = papers_tracker_image_link(png_path, doc=PAPER_NGUYEN_DOC)
    if update_docs and PAPER_NGUYEN_DOC.exists():
        text = PAPER_NGUYEN_DOC.read_text()
        text = _replace_marker_in(
            text,
            "caption-3",
            _caption_block(caption, PAPER_NGUYEN_2_3_MANIFEST),
            doc=PAPER_NGUYEN_DOC,
        )
        text = _set_papers_tracker_image_link(
            text,
            alt=PAPER_NGUYEN_2_3_REPLICATION_ALT,
            link=link,
            doc=PAPER_NGUYEN_DOC,
        )
        PAPER_NGUYEN_DOC.write_text(text)
    return {
        "png": str(png_path),
        "png_link": link,
        "manifest": PAPER_NGUYEN_2_3_MANIFEST,
        "caption": caption,
        "doc": str(PAPER_NGUYEN_DOC),
    }


def promote_nguyen_4(
    *,
    manifest: dict[str, Any],
    png_path: Path,
    update_docs: bool = True,
) -> dict[str, str]:
    """Refresh Fig 4 caption + replication image link in ``figures/nguyen/replications.md``."""
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    link = papers_tracker_image_link(png_path, doc=PAPER_NGUYEN_DOC)
    if update_docs and PAPER_NGUYEN_DOC.exists():
        text = PAPER_NGUYEN_DOC.read_text()
        text = _replace_marker_in(
            text,
            "caption-4",
            _caption_block(caption, PAPER_NGUYEN_4_MANIFEST),
            doc=PAPER_NGUYEN_DOC,
        )
        text = _set_papers_tracker_image_link(
            text,
            alt=PAPER_NGUYEN_4_REPLICATION_ALT,
            link=link,
            doc=PAPER_NGUYEN_DOC,
        )
        # Refresh Fig 4 Status line (any prior Open/Pass wording after caption-4).
        import re as _re
        _pass = bool(manifest.get("gates", {}).get("pass"))
        _status = (
            f"**Status:** Pass — see manifest gates (`{png_path.name}`)."
            if _pass
            else f"**Status:** Open — see manifest gates (`{png_path.name}`)."
        )
        text = _re.sub(
            r"(<!-- caption-4:end -->\s*\n)\*\*Status:\*\*[^\n]+",
            r"\1" + _status,
            text,
            count=1,
        )
        # Keep summary-table Status column in sync for Fig 4 row.
        text = _re.sub(
            r"(\| Fig 4 — training reward \+ episode length \|[^|]+\|[^|]+\|[^|]+\| )(Open|Pass)( \|)",
            r"\1" + ("Pass" if _pass else "Open") + r"\3",
            text,
            count=1,
        )
        text = text.replace(
            "*Not yet generated.* Target: `figures/nguyen/images/4/training_reward_length.png`",
            f"![Replication Fig 4]({link})",
        )
        PAPER_NGUYEN_DOC.write_text(text)
    return {
        "png": str(png_path),
        "png_link": link,
        "manifest": PAPER_NGUYEN_4_MANIFEST,
        "caption": caption,
        "doc": str(PAPER_NGUYEN_DOC),
    }


def resolve_ravivarapu_doc(checkout: Path | None = None) -> Path:
    root = main_checkout_root(checkout or CHECKOUT_ROOT)
    return root / "figures" / "ravivarapu" / "replications.md"


PAPER_RAVIVARAPU_DOC = resolve_ravivarapu_doc()


def promote_ravivarapu_4a(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    del update_docs
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    link = repo_rel_posix(png_path)
    if PAPER_RAVIVARAPU_DOC.exists():
        text = PAPER_RAVIVARAPU_DOC.read_text()
        if "<!-- caption-4a:start -->" in text:
            text = _replace_marker(
                text,
                "caption-4a",
                _caption_block(caption, "artifacts/figures/papers/ravivarapu/4/manifest_4a.json"),
            )
        text = text.replace(
            "*Not yet generated.* Target: `figures/ravivarapu/images/4a/training_psd.png`",
            f"![Replication Fig 4a](images/4a/{png_path.name})",
        )
        text = text.replace(
            "**Status:** Open — needs SEA-DBS trainer + Baseline ablation.",
            f"**Status:** {'Pass' if manifest.get('gates', {}).get('pass') else 'Open'} — see manifest gates.",
        )
        PAPER_RAVIVARAPU_DOC.write_text(text)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}


def promote_ravivarapu_4b(*, manifest: dict[str, Any], png_path: Path, update_docs: bool = True) -> dict[str, str]:
    del update_docs
    caption = manifest.get("caption") or "see manifest"
    if manifest.get("png_version") is not None:
        caption = f"{caption} (v{manifest['png_version']})"
    if PAPER_RAVIVARAPU_DOC.exists():
        text = PAPER_RAVIVARAPU_DOC.read_text()
        if "<!-- caption-4b:start -->" in text:
            text = _replace_marker(
                text,
                "caption-4b",
                _caption_block(caption, "artifacts/figures/papers/ravivarapu/4/manifest_4b.json"),
            )
        text = text.replace(
            "*Not yet generated.* Target: `figures/ravivarapu/images/4b/training_reward.png`",
            f"![Replication Fig 4b](images/4b/{png_path.name})",
        )
        text = text.replace(
            "**Status:** Open — pair with Fig 4a locked run.",
            f"**Status:** {'Pass' if manifest.get('gates', {}).get('pass') else 'Open'} — see manifest gates.",
        )
        PAPER_RAVIVARAPU_DOC.write_text(text)
    return {"png": str(png_path), "caption": caption, "doc": str(PAPER_RAVIVARAPU_DOC)}
