#!/usr/bin/env python3
"""Refresh replication image paths in knowledge-base ``reports/3.md``.

Report 3 gallery embeds use **co-located copies** under
``reports/images/gallery/`` (``./images/gallery/rep_*.png``) so Obsidian
resolves them from the note folder regardless of vault root. Ship PNGs under
``figures/<paper>/images/`` are still refreshed for trackers.

``sync_report3_gallery_ships()`` copies tracker versioned PNGs into ship paths.
``sync_report3_gallery_embeds()`` copies bytes beside the report.
``update_report3()`` rewrites gallery ``![Rep …](…)`` hrefs.
``sync_report3_panel_notes()`` ensures claim, status, quoted caption, and brief
gate spec blocks for each gallery panel (marker blocks ``report3-*`` /
``paper-caption-rep-*``).

Panel ``plot.py`` runs enable gallery refresh with ``--update-report`` (see
``resume_cli``).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path

_PROMOTE = Path(__file__).resolve().parent / "promote.py"
_spec = importlib.util.spec_from_file_location("figure_promote", _PROMOTE)
assert _spec and _spec.loader
_promote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_promote)

REPO_ROOT = _promote.REPO_ROOT

# Report 3 alt text -> (paper dir under figures/, stable ship path under paper/).
REPORT3_SHIP_PATHS: dict[str, tuple[str, str]] = {
    "Rep 1b": ("mehregan", "images/1b/gpi_psd.png"),
    "Rep 2a": ("mehregan", "images/2a/beta_power.png"),
    "Rep 2b": ("mehregan", "images/2b/error_index.png"),
    "Rep 4a": ("mehregan", "images/4a/training_beta.png"),
    "Rep 4b": ("mehregan", "images/4b/training_fig4b.png"),
    "Rep 5a": ("mehregan", "images/5a/efficacy_45hz.png"),
    "Rep 5b": ("mehregan", "images/5b/efficacy_30hz.png"),
    "Rep 6a": ("mehregan", "images/6a/ptq_qat_45hz.png"),
    "Rep 6b": ("mehregan", "images/6b/ptq_qat_30hz.png"),
    "Rep N3": ("nguyen", "images/3/alpha_beta_dist.png"),
    "Rep N4": ("nguyen", "images/4/training_reward_length.png"),
    "Rep R4a": ("ravivarapu", "images/4a/training_psd.png"),
}

# Co-located gallery filenames (beside ``reports/3.md``).
REPORT3_GALLERY_FILES: dict[str, str] = {
    "Rep 1b": "rep_1b.png",
    "Rep 2a": "rep_2a.png",
    "Rep 2b": "rep_2b.png",
    "Rep 4a": "rep_4a.png",
    "Rep 4b": "rep_4b.png",
    "Rep 5a": "rep_5a.png",
    "Rep 5b": "rep_5b.png",
    "Rep 6a": "rep_6a.png",
    "Rep 6b": "rep_6b.png",
    "Rep N3": "rep_n3.png",
    "Rep N4": "rep_n4.png",
    "Rep R4a": "rep_r4a.png",
}

# Report 3 alt -> tracker ``![…](images/…)`` alt substring for the versioned source.
REPORT3_TRACKER_ALTS: dict[str, tuple[str, str]] = {
    "Rep 1b": ("mehregan", "Replication Fig 1b"),
    "Rep 2a": ("mehregan", "Replication Fig 2a"),
    "Rep 2b": ("mehregan", "Replication Fig 2b"),
    "Rep 4a": ("mehregan", "Replication Fig 4a"),
    "Rep 4b": ("mehregan", "Replication Fig 4b combined"),
    "Rep 5a": ("mehregan", "Replication Fig 5a"),
    "Rep 5b": ("mehregan", "Replication Fig 5b"),
    "Rep 6a": ("mehregan", "Replication Fig 6a"),
    "Rep 6b": ("mehregan", "Replication Fig 6b"),
    "Rep N3": ("nguyen", "Replication Fig 3"),
    "Rep N4": ("nguyen", "Replication Fig 4 — latest"),
    "Rep R4a": ("ravivarapu", "Replication Fig 4a"),
}

_IMAGE_LINK_RE = re.compile(r"!\[([^\]]*)\]\((images/[^)]+)\)")

REPORT3_PAPER_CITE: dict[str, str] = {
    "mehregan": "Mehregan et al.",
    "nguyen": "Nguyen et al.",
    "ravivarapu": "Ravivarapu et al.",
}


def _paper_cite_for_rep(report_alt: str) -> str:
    paper, _ = REPORT3_SHIP_PATHS[report_alt]
    return REPORT3_PAPER_CITE[paper]

# Per-panel gallery text (claim, status, quoted caption, brief gates).
REPORT3_PANEL_META: dict[str, dict[str, str]] = {
    "Rep 1b": {
        "claim": (
            "PD elevates GPi beta vs healthy; **130 Hz** cDBS suppresses that elevation."
        ),
        "status": "Pass: ordering and beta-peak shape match (seeds 0–9 mean).",
        "caption": (
            "Oscillatory activity of model neurons in the GPi under the healthy "
            "state, PD state, and PD state with $130\\text{Hz}$ conventional stimulation."
        ),
        "gates": (
            "PD > healthy on beta; 130 Hz cDBS < untreated PD; suppression ratio "
            "and beta levels within band of digitized paper."
        ),
    },
    "Rep 2a": {
        "claim": (
            "After cDBS onset at **2 s**, treated (blue) falls and stays below "
            "untreated PD (red); shared pre-stim baseline."
        ),
        "status": (
            "Pass: blue-below-red after $t=2$, shared 0–2 s baseline, dense "
            "trailing protocol."
        ),
        "caption": "cDBS effects on beta power of GPi neurons",
        "gates": (
            "Shared pre-onset baseline; treated < untreated after onset; late "
            "ratio and suppression drop vs digitized paper."
        ),
    },
    "Rep 2b": {
        "claim": (
            "Same timing as 2a; Error Index (thalamic spike-timing) lower under "
            "cDBS after onset."
        ),
        "status": (
            "Pass: hybrid So-style SMC → thalamus on the Kumaravelu body "
            "(documented plant convention)."
        ),
        "caption": "cDBS effects on error.",
        "gates": (
            "Same timing gates as Fig. 2a on the Error Index biomarker "
            "(pre-onset, late ordering, digitization ratios)."
        ),
    },
    "Rep 4a": {
        "claim": (
            "Noisy early β, sharp mid-run drop, lower late plateau (45 Hz DDPG "
            "training)."
        ),
        "status": (
            "Revisit in progress: pair with 4b; softmax τ 3→1.4 to avoid "
            "pattern-0 collapse (locked v18 was τ→1.0)."
        ),
        "caption": (
            "The average power of the beta frequency band for each step during "
            "training of the $45\\text{Hz}$ average frequency model."
        ),
        "gates": (
            "Training trend down; mid-run fade and late/early ratio vs digitized "
            "paper; drop magnitude ≥ 70% of paper drop."
        ),
    },
    "Rep 4b": {
        "claim": (
            "Over episodes 0–8, reward rises toward zero while episode-mean β falls."
        ),
        "status": (
            "Revisit in progress: match digitized late PSD floor (~0.37, above "
            "β_t) so reward approaches 0 from below; paired with latest Fig. 4a series."
        ),
        "caption": (
            "Beta power in GPi during each episode of training of the $45\\text{Hz}$ "
            "(top) and accumulated reward of each training episode for the "
            "$45\\text{Hz}$ (bottom)."
        ),
        "gates": (
            "Reward rises from negative early values; episode-mean PSD falls; rise "
            "timing and PSD drop ratio vs digitized paper; late PSD ≥ 0.35 and "
            "within 15% of paper; late reward in (−10, 2]."
        ),
    },
    "Rep 5a": {
        "claim": (
            "Trained irregular pattern suppresses β vs no stim, but stays above "
            "periodic 45 Hz; **130 Hz** lowest."
        ),
        "status": "Pass: `skip_regular` alphabet; four-series trailing eval.",
        "caption": "Performance of the fully trained model at $45 \\text{ Hz}$.",
        "gates": (
            "Shared baseline; ordering 130 Hz < trained < no stim < periodic 45 Hz; "
            "late trained/no-stim and periodic/no-stim ratios vs digitized paper."
        ),
    },
    "Rep 5b": {
        "claim": (
            "Periodic 30 Hz *elevates* β; trained irregular pattern goes **below** "
            "both no stim and periodic."
        ),
        "status": "Pass: burst-pattern alphabet (mean rate still 30 Hz).",
        "caption": "Performance of the fully trained model at $30 \\text{ Hz}$.",
        "gates": (
            "Periodic elevates vs no stim; trained below both; late ratios vs "
            "digitized paper."
        ),
    },
    "Rep 6a": {
        "claim": (
            "fp32 / PTQ fp16 / PTQ int8 suppress in a shared band after onset; "
            "**QAT fails** (stays elevated)."
        ),
        "status": (
            "Pass: trailing eval; PTQ tracks fp32; QAT elevated vs digitized "
            "paper band."
        ),
        "caption": "Performance PTQ and QAT at $45 \\text{ Hz}$.",
        "gates": (
            "fp32 suppresses; PTQ fp16/int8 track fp32 and stay distinct; QAT elevated "
            "vs fp32 and near baseline; digitization-backed level gates for all four "
            "series."
        ),
    },
    "Rep 6b": {
        "claim": "Same four-series story as 6a at **30 Hz**.",
        "status": (
            "Pass: tier PTQ; weak QAT lock for failed-QAT claim."
        ),
        "caption": "Performance PTQ and QAT at $30 \\text{ Hz}$.",
        "gates": (
            "Same qualitative split as 6a at 30 Hz: PTQ tracks fp32; QAT elevated; "
            "digitization level gates."
        ),
    },
    "Rep N3": {
        "claim": "PD On α–β power higher than PD Off.",
        "status": (
            "Pass: 500 × 100 ms samples; 7–35 Hz α–β index (not Mehregan's "
            "13–35 Hz $P_\\beta$)."
        ),
        "caption": (
            "Distribution of GPi $\\alpha$-$\\beta$ oscillation power values for PD "
            "and no-PD; each sample (scatter) and summarized as a boxplot."
        ),
        "gates": (
            "PD On > PD Off; ordering and mean ratio vs digitized paper readout."
        ),
    },
    "Rep N4": {
        "claim": (
            "High early variance; reward rises and episode length falls over 500 "
            "episodes."
        ),
        "status": (
            "**In progress:** reward-plateau **shape** gates still fail."
        ),
        "caption": (
            "Training rewards and episode lengths over 500 episodes, showing "
            "progression from exploration to optimization."
        ),
        "gates": (
            "Reward: scale + late > early (shape pass); reward plateau ep 100–450 still "
            "fails. Length: decreases + late ≤ 12 (shape pass); mid glide / post-100 "
            "plateau still fail."
        ),
    },
    "Rep R4a": {
        "claim": (
            "SEA-DBS shows stronger, more consistent β suppression over episodes than "
            "Baseline DDPG."
        ),
        "status": "**In progress:** **shape** gates still fail; digitized paper overlay on the same axes.",
        "caption": "PSD over training episodes (SEA-DBS vs Baseline).",
        "gates": (
            "SEA-DBS declines steeper than baseline; late SEA below baseline; "
            "trajectory shape + digitization trajectory gates (`shape_pass` and full "
            "`pass`)."
        ),
    },
}


def _panel_slug(report_alt: str) -> str:
    return report_alt.lower().replace(" ", "-")


def _caption_marker(report_alt: str) -> str:
    return f"paper-caption-{_panel_slug(report_alt)}"


def _quoted_caption_line(caption: str) -> str:
    return f'> "{caption}"'


def _caption_block(report_alt: str) -> str:
    caption = _quoted_caption_line(REPORT3_PANEL_META[report_alt]["caption"])
    cite = f"*{_paper_cite_for_rep(report_alt)}*"
    marker = _caption_marker(report_alt)
    return f"<!-- {marker}:start -->\n{caption}\n> {cite}\n<!-- {marker}:end -->\n"


def _caption_block_end_pos(text: str, report_alt: str) -> int | None:
    marker = _caption_marker(report_alt)
    end_token = f"<!-- {marker}:end -->"
    idx = text.find(end_token)
    if idx < 0:
        return None
    return idx + len(end_token)


def _claim_block(report_alt: str) -> str:
    meta = REPORT3_PANEL_META[report_alt]
    slug = _panel_slug(report_alt)
    marker = f"report3-claim-{slug}"
    return (
        f"<!-- {marker}:start -->\n"
        f"**Paper claim:** {meta['claim']}\n"
        f"<!-- {marker}:end -->"
    )


def _status_block(report_alt: str) -> str:
    meta = REPORT3_PANEL_META[report_alt]
    slug = _panel_slug(report_alt)
    marker = f"report3-status-{slug}"
    return (
        f"<!-- {marker}:start -->\n"
        f"**Status:** {meta['status']}\n"
        f"<!-- {marker}:end -->"
    )


def _gates_block(report_alt: str) -> str:
    meta = REPORT3_PANEL_META[report_alt]
    slug = _panel_slug(report_alt)
    marker = f"report3-gates-{slug}"
    return (
        f"<!-- {marker}:start -->\n"
        f"**Gates example:** {meta['gates']}\n"
        f"<!-- {marker}:end -->"
    )


def _sync_marked_block(
    text: str,
    marker: str,
    desired_body: str,
    *,
    insert_at: int | None = None,
) -> tuple[str, bool]:
    """Replace or insert a ``<!-- marker:start -->…<!-- marker:end -->`` block."""
    block = f"<!-- {marker}:start -->\n{desired_body}\n<!-- {marker}:end -->"
    marker_re = re.compile(
        rf"<!-- {re.escape(marker)}:start -->.*?<!-- {re.escape(marker)}:end -->",
        re.DOTALL,
    )
    existing = marker_re.search(text)
    if existing:
        if existing.group(0).strip() == block.strip():
            return text, False
        return marker_re.sub(lambda _m: block, text, count=1), True
    if insert_at is None:
        return text, False
    return text[:insert_at] + block + "\n\n" + text[insert_at:], True


def sync_report3_panel_notes(*, dry_run: bool = False, verbose: bool = True) -> dict[str, object]:
    """Refresh claim, status, quoted captions, and brief gate specs in Report 3."""
    path = report3_path()
    text = path.read_text(encoding="utf-8")
    changes: list[str] = []

    for report_alt in REPORT3_PANEL_META:
        slug = _panel_slug(report_alt)
        embed_re = re.compile(rf"!\[{re.escape(report_alt)}\]\([^)]+\)\n?")
        embed_match = embed_re.search(text)
        if embed_match is None:
            if verbose:
                print(f"report3-panel: skip {report_alt} (no embed)", flush=True)
            continue

        embed_start = embed_match.start()
        text, changed = _sync_marked_block(
            text,
            f"report3-claim-{slug}",
            f"**Paper claim:** {REPORT3_PANEL_META[report_alt]['claim']}",
            insert_at=embed_start,
        )
        if changed:
            changes.append(f"claim:{report_alt}")
            embed_match = embed_re.search(text)
            embed_start = embed_match.start() if embed_match else embed_start

        text, changed = _sync_marked_block(
            text,
            f"report3-status-{slug}",
            f"**Status:** {REPORT3_PANEL_META[report_alt]['status']}",
            insert_at=embed_start,
        )
        if changed:
            changes.append(f"status:{report_alt}")
            embed_match = embed_re.search(text)
            embed_start = embed_match.start() if embed_match else embed_start

        caption_marker = _caption_marker(report_alt)
        caption_re = re.compile(
            rf"<!-- {re.escape(caption_marker)}:start -->.*?<!-- {re.escape(caption_marker)}:end -->",
            re.DOTALL,
        )
        caption_match = caption_re.search(text)
        caption_block = _caption_block(report_alt)
        if caption_match:
            if caption_match.group(0).strip() != caption_block.strip():
                text = caption_re.sub(lambda _m: caption_block, text, count=1)
                changes.append(f"caption:{report_alt}")
        else:
            insert_at = embed_match.end() if embed_match else embed_start
            text = text[:insert_at] + "\n" + caption_block + text[insert_at:]
            changes.append(f"caption:{report_alt}")

        gates_insert = _caption_block_end_pos(text, report_alt)
        text, changed = _sync_marked_block(
            text,
            f"report3-gates-{slug}",
            f"**Gates example:** {REPORT3_PANEL_META[report_alt]['gates']}",
            insert_at=gates_insert,
        )
        if changed:
            changes.append(f"gates:{report_alt}")

    if changes and not dry_run:
        text = re.sub(r"\n{3,}", "\n\n", text)
        path.write_text(text, encoding="utf-8")

    if verbose:
        for change in changes:
            print(f"report3-panel: {change}", flush=True)
        if not changes:
            print("report3-panel: already up to date", flush=True)

    return {"path": str(path), "n_changes": len(changes), "changes": changes}


def sync_report3_paper_captions(*, dry_run: bool = False, verbose: bool = True) -> dict[str, object]:
    """Backward-compatible alias: refreshes all panel note blocks."""
    return sync_report3_panel_notes(dry_run=dry_run, verbose=verbose)


def report3_path() -> Path:
    for base in (
        Path.home() / "knowledge-base",
        Path.home() / "Insync" / "knowledge-base",
    ):
        candidate = (
            base
            / "bme"
            / "brain-stimulation"
            / "rl-adaptive-dbs"
            / "reports"
            / "3.md"
        )
        if candidate.is_file():
            return candidate
    msg = "could not locate knowledge-base reports/3.md"
    raise FileNotFoundError(msg)


def _tracker_link(tracker_text: str, alt_substring: str) -> str | None:
    for match in _IMAGE_LINK_RE.finditer(tracker_text):
        if alt_substring in match.group(1):
            return match.group(2)
    return None


def report3_gallery_dir() -> Path:
    return report3_path().parent / "images" / "gallery"


def _report_href(report_alt: str) -> str:
    return f"./images/gallery/{REPORT3_GALLERY_FILES[report_alt]}"


def _ship_path(paper: str, ship_rel: str) -> Path:
    return REPO_ROOT / "figures" / paper / ship_rel


def _versioned_source(report_alt: str) -> Path | None:
    paper, tracker_alt = REPORT3_TRACKER_ALTS[report_alt]
    tracker_doc = REPO_ROOT / "figures" / paper / "replications.md"
    if not tracker_doc.is_file():
        return None
    tracker_rel = _tracker_link(tracker_doc.read_text(encoding="utf-8"), tracker_alt)
    if tracker_rel is None:
        return None
    source = REPO_ROOT / "figures" / paper / tracker_rel
    return source if source.is_file() else None


def sync_report3_gallery_ships(*, verbose: bool = True) -> list[dict[str, str]]:
    """Copy tracker-linked versioned PNGs into Report 3 stable ship paths."""
    synced: list[dict[str, str]] = []
    for report_alt, (paper, ship_rel) in REPORT3_SHIP_PATHS.items():
        source = _versioned_source(report_alt)
        ship_repo_rel = f"figures/{paper}/{ship_rel}"
        if source is None:
            if verbose:
                print(f"report3-ship: skip {report_alt} (no tracker source)", flush=True)
            continue
        dest = _promote.materialize_ship_png(source, ship_repo_rel)
        row = {
            "alt": report_alt,
            "source": str(source),
            "ship": str(dest),
        }
        synced.append(row)
        if verbose:
            print(
                f"report3-ship: {report_alt}: {source.name} -> {ship_rel}",
                flush=True,
            )
    return synced


def sync_report3_gallery_embeds(*, verbose: bool = True) -> list[dict[str, str]]:
    """Copy gallery PNG bytes into ``reports/images/gallery/`` beside Report 3."""
    gallery_dir = report3_gallery_dir()
    gallery_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for report_alt, (paper, ship_rel) in REPORT3_SHIP_PATHS.items():
        gallery_name = REPORT3_GALLERY_FILES[report_alt]
        ship = _ship_path(paper, ship_rel)
        source = ship if ship.is_file() else _versioned_source(report_alt)
        if source is None:
            if verbose:
                print(f"report3-gallery: skip {report_alt} (no source)", flush=True)
            continue
        dest = gallery_dir / gallery_name
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        row = {"alt": report_alt, "source": str(source), "gallery": str(dest)}
        copied.append(row)
        if verbose:
            print(
                f"report3-gallery: {report_alt}: {source.name} -> images/gallery/{gallery_name}",
                flush=True,
            )
    return copied


def build_report3_replacements() -> dict[str, str]:
    """Map Report 3 image alt -> co-located ``./images/gallery/...`` href."""
    return {report_alt: _report_href(report_alt) for report_alt in REPORT3_GALLERY_FILES}


def update_report3(*, dry_run: bool = False, verbose: bool = True) -> dict[str, object]:
    path = report3_path()
    text = path.read_text(encoding="utf-8")
    replacements = build_report3_replacements()
    changes: list[dict[str, str]] = []

    for report_alt, new_href in replacements.items():
        pattern = re.compile(
            rf"(!\[{re.escape(report_alt)}\]\()([^)]+)(\))",
        )
        match = pattern.search(text)
        if match is None:
            continue
        old_href = match.group(2)
        if old_href == new_href:
            continue
        changes.append({"alt": report_alt, "old": old_href, "new": new_href})
        text = pattern.sub(rf"\1{new_href}\3", text, count=1)

    if changes and not dry_run:
        path.write_text(text, encoding="utf-8")

    if verbose:
        for row in changes:
            print(f"report3: {row['alt']}: {row['old']} -> {row['new']}", flush=True)
        if not changes:
            print("report3: already up to date", flush=True)

    return {"path": str(path), "n_changes": len(changes), "changes": changes}


def refresh_report3_gallery(*, dry_run: bool = False, verbose: bool = True) -> dict[str, object]:
    """Sync ship + co-located gallery PNG bytes, then rewrite Report 3 embed hrefs."""
    if dry_run:
        ships = [
            {
                "alt": alt,
                "source": str(_versioned_source(alt) or ""),
                "ship": f"figures/{paper}/{rel}",
            }
            for alt, (paper, rel) in REPORT3_SHIP_PATHS.items()
        ]
        gallery = [
            {
                "alt": alt,
                "gallery": str(report3_gallery_dir() / REPORT3_GALLERY_FILES[alt]),
            }
            for alt in REPORT3_GALLERY_FILES
        ]
        report = update_report3(dry_run=True, verbose=verbose)
        captions = sync_report3_panel_notes(dry_run=True, verbose=verbose)
        return {"ships": ships, "gallery": gallery, "captions": captions, **report}
    ships = sync_report3_gallery_ships(verbose=verbose)
    gallery = sync_report3_gallery_embeds(verbose=verbose)
    report = update_report3(dry_run=False, verbose=verbose)
    captions = sync_report3_panel_notes(dry_run=False, verbose=verbose)
    return {"ships": ships, "gallery": gallery, "captions": captions, **report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--ships-only",
        action="store_true",
        help="Only materialize ship PNGs; do not rewrite reports/3.md.",
    )
    parser.add_argument(
        "--captions-only",
        action="store_true",
        help="Only refresh panel notes (claim, status, captions, gates) in reports/3.md.",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args(argv)
    verbose = not args.quiet
    if args.captions_only:
        sync_report3_panel_notes(dry_run=args.dry_run, verbose=verbose)
    elif args.ships_only:
        sync_report3_gallery_ships(verbose=verbose)
        sync_report3_gallery_embeds(verbose=verbose)
    elif args.dry_run:
        refresh_report3_gallery(dry_run=True, verbose=verbose)
    else:
        refresh_report3_gallery(dry_run=False, verbose=verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
