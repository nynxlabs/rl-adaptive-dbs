"""Build a WebPlotDigitizer project from a PIL rough extract.

Writes a ``.wpd.json`` (axes + points) and optionally a ``.tar`` project
that also embeds the paper PNG — WPD's one-click Load Project format
(``info.json`` + ``wpd.json`` + image), so you do not need a separate
Load Image step.

WPD has no remote load API; this is the automation we can do without
driving a browser. Clean up points in WPD, then Save / export CSV and
``normalize_wpd.py`` as usual.

Requires a digitization fig config (axis box + ranges) and a schema JSON
that includes per-series ``xy`` traces (``extract_pil.py`` writes these).

Usage:

    uv run python scripts/digitization/pil_to_wpd.py \\
        --curves artifacts/.../curves_pil.json \\
        --config scripts/digitization/figs/mehregan_fig1b.json \\
        --png figures/mehregan/images/1b/paper.png \\
        --out artifacts/.../fig1b_pil_seed.wpd.json \\
        --tar artifacts/.../fig1b_pil_seed.wpd.tar
"""
from __future__ import annotations

import argparse
import json
import tarfile
from io import BytesIO
from pathlib import Path

# Canonical schema key -> WPD dataset display name
DEFAULT_NAME_MAP = {
    "pd": "PD no Treatment",
    "healthy": "Healthy Control",
    "pd_130hz": "PD 130 Hz Treatment",
}

# Approximate legend colors (RGBA) for the three Mehregan 1b series
DEFAULT_COLORS = {
    "pd": [246, 194, 153, 255],
    "healthy": [15, 13, 14, 255],
    "pd_130hz": [81, 73, 120, 255],
}


def data_to_pixel(
    x: float,
    y: float,
    box: tuple[int, int, int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[float, float]:
    """Map data coords → image pixel coords (WPD's x/y on points)."""
    top, bottom, left, right = box
    w = right - left
    h = bottom - top
    px = left + (x - x_range[0]) / (x_range[1] - x_range[0]) * w
    py = top + (y_range[1] - y) / (y_range[1] - y_range[0]) * h
    return float(px), float(py)


def build_project(
    curves: dict,
    config: dict,
    name_map: dict[str, str],
    colors: dict[str, list[int]],
) -> dict:
    panel = config["panel"]
    box = tuple(panel["box"])
    x_range = tuple(panel["x_range"])
    y_range = tuple(panel["y_range"])
    top, bottom, left, right = box

    axes = {
        "name": "XY",
        "type": "XYAxes",
        "isLogX": False,
        "isLogY": False,
        "noRotation": True,
        "calibrationPoints": [
            {"px": float(left), "py": float(bottom), "dx": str(x_range[0]), "dy": str(y_range[0]), "dz": None},
            {"px": float(right), "py": float(bottom), "dx": str(x_range[1]), "dy": str(y_range[0]), "dz": None},
            {"px": float(left), "py": float(bottom), "dx": str(x_range[0]), "dy": str(y_range[0]), "dz": None},
            {"px": float(left), "py": float(top), "dx": str(x_range[0]), "dy": str(y_range[1]), "dz": None},
        ],
    }

    datasets = []
    for key, stats in curves.get("series", {}).items():
        xy = stats.get("xy")
        if not xy:
            raise ValueError(
                f"series '{key}' has no xy traces — re-run extract_pil.py "
                "(newer builds embed xy in the schema JSON)"
            )
        xs, ys = xy["x"], xy["y"]
        display = name_map.get(key, key)
        color = stats.get("color_rgba") or colors.get(key, [255, 0, 0, 255])
        pts = []
        for xv, yv in zip(xs, ys):
            px, py = data_to_pixel(float(xv), float(yv), box, x_range, y_range)
            pts.append({"x": px, "y": py, "value": [float(xv), float(yv)]})
        datasets.append(
            {
                "name": display,
                "axesName": "XY",
                "colorRGB": color,
                "metadataKeys": [],
                "data": pts,
                "autoDetectionData": None,
            }
        )

    return {
        "version": [4, 2],
        "axesColl": [axes],
        "datasetColl": datasets,
        "measurementColl": [],
        "_provenance": {
            "method": "pil-seeded-wpd",
            "source_curves": curves.get("source_png") or curves.get("source"),
            "source_method": curves.get("method"),
            "config_figure": config.get("figure"),
            "note": "Load the paper PNG first in WPD, then File → Load Project on this JSON. Edit points, then Save/Export.",
        },
    }


def write_wpd_tar(tar_path: Path, project_name: str, wpd_json: str, png_path: Path) -> None:
    """Pack a WPD .tar project (image + wpd.json + info.json).

    Layout matches automeris-io/WebPlotDigitizer ``saveResume._writeAndDownloadTar``:
        <projectName>/info.json
        <projectName>/wpd.json
        <projectName>/<imageFileName>
    """
    image_name = png_path.name
    # Prefer a stable png name inside the archive
    if png_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".pdf"}:
        image_name = "paper.png"
    info = json.dumps(
        {"version": [4, 0], "json": "wpd.json", "images": [image_name]},
        indent=2,
    )
    png_bytes = png_path.read_bytes()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        def _add(name: str, data: bytes) -> None:
            info_obj = tarfile.TarInfo(name=f"{project_name}/{name}")
            info_obj.size = len(data)
            tar.addfile(info_obj, BytesIO(data))

        _add("info.json", info.encode("utf-8"))
        _add("wpd.json", wpd_json.encode("utf-8"))
        _add(image_name, png_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curves", type=Path, required=True, help="curves_pil.json (with xy)")
    ap.add_argument("--config", type=Path, required=True, help="figs/<panel>.json")
    ap.add_argument("--out", type=Path, required=True, help="output .wpd.json")
    ap.add_argument("--png", type=Path, help="paper PNG (required with --tar)")
    ap.add_argument("--tar", type=Path, help="also write a one-click WPD .tar project")
    ap.add_argument("--project-name", default=None,
                    help="folder name inside the .tar (default: --tar stem)")
    args = ap.parse_args()

    if args.tar and not args.png:
        ap.error("--tar requires --png so the image can be embedded")

    curves = json.loads(args.curves.read_text())
    config = json.loads(args.config.read_text())
    project = build_project(curves, config, DEFAULT_NAME_MAP, DEFAULT_COLORS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance = project.pop("_provenance", None)
    wpd_text = json.dumps(project, indent=2) + "\n"
    args.out.write_text(wpd_text)
    if provenance is not None:
        meta = args.out.with_suffix(".meta.json")
        meta.write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"wrote {meta}")
    print(f"wrote {args.out}: {len(project['datasetColl'])} datasets")

    if args.tar:
        png = args.png.resolve()
        if not png.is_file():
            raise SystemExit(f"PNG not found: {png}")
        project_name = args.project_name or args.tar.stem
        write_wpd_tar(args.tar, project_name, wpd_text, png)
        print(f"wrote {args.tar} (one-click Load Project in WPD)")
        print("In WPD: File → Load Project → pick the .tar → edit → Save/Export")
    else:
        print("In WPD: Load Image (paper.png) → File → Load Project (this JSON) → edit")


if __name__ == "__main__":
    main()
