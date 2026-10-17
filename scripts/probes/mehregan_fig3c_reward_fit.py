#!/usr/bin/env python3
"""Trace Mehregan Fig 3c (reward vs average beta power) and fit Eq. (8) readings.

Fig 3c is a black curve on white, x 0–1, y −60–20. The probe finds the plot frame,
traces the curve column by column (intensity-weighted dark pixels), writes the
trace as a thin ``curves`` JSON, and scores two readings of Eq. (8) below
$\\beta_t = 0.35$:

- ``square``: $+((\\beta_t - s) \\cdot 10)^2$ (signed square, smooth through 0)
- ``linear``: $(\\beta_t - s) \\cdot 10$ (printed branch with its sign flipped)

Both share the printed quadratic penalty above $\\beta_t$. It also fits
$a (\\beta_t - s)^p$ below threshold by grid search.

Run (image = the paper's Figure 3 composite, panel c bottom right):
  uv run python scripts/probes/mehregan_fig3c_reward_fit.py FIG3.png \\
      --out artifacts/figures/papers/mehregan/3c/paper_digitization/curves_pil.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BETA_T = 0.35
X_RANGE = (0.0, 1.0)
Y_RANGE = (-60.0, 20.0)
# Panel c crop inside the 684×573 Figure 3 composite (left, top, right, bottom).
DEFAULT_CROP = (320, 395, 684, 573)


def find_frame(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Frame lines = columns / rows with long runs of dark pixels."""
    dark = gray < 200
    cols = [i for i, v in enumerate(dark.sum(0)) if v > 80]
    rows = [i for i, v in enumerate(dark.sum(1)) if v > 100]
    return cols[0], cols[-1], rows[0], rows[1]


def trace_curve(gray: np.ndarray, frame: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    left, right, top, bottom = frame
    xs: list[float] = []
    ys: list[float] = []
    for c in range(left + 2, right - 1):
        col = gray[top + 2 : bottom - 1, c]
        idx = np.where(col < 110)[0]
        if idx.size == 0:
            continue
        w = 255.0 - col[idx]
        r = float((idx * w).sum() / w.sum()) + top + 2
        xs.append(X_RANGE[0] + (c - left) / (right - left) * (X_RANGE[1] - X_RANGE[0]))
        ys.append(Y_RANGE[1] - (r - top) / (bottom - top) * (Y_RANGE[1] - Y_RANGE[0]))
    return np.asarray(xs), np.asarray(ys)


def reading(x: np.ndarray, kind: str) -> np.ndarray:
    delta = (x - BETA_T) * 10.0
    above = -(delta**2)
    below = delta**2 if kind == "square" else -delta
    return np.where(x < BETA_T, below, above)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", type=Path)
    ap.add_argument("--crop", type=int, nargs=4, default=DEFAULT_CROP)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gray = np.asarray(Image.open(args.image).convert("L"), dtype=float)
    l, t, r, b = args.crop
    sub = gray[t:b, l:r]
    frame = find_frame(sub)
    x, y = trace_curve(sub, frame)
    below = x < BETA_T

    result: dict[str, object] = {"n_points": int(x.size), "frame_px": frame}
    for kind in ("square", "linear"):
        m = reading(x, kind)
        result[kind] = {
            "rmse_all": float(np.sqrt(np.mean((y - m) ** 2))),
            "rmse_below": float(np.sqrt(np.mean((y[below] - m[below]) ** 2))),
            "r_at_0": float(reading(np.array([0.0]), kind)[0]),
        }
    best = (np.inf, 0.0, 0.0)
    for p in np.linspace(0.5, 3.0, 51):
        for a in np.linspace(1.0, 200.0, 400):
            e = float(np.mean((y[below] - a * (BETA_T - x[below]) ** p) ** 2))
            if e < best[0]:
                best = (e, float(a), float(p))
    result["power_fit_below"] = {"a": best[1], "p": best[2], "rmse": float(np.sqrt(best[0]))}
    result["digitized_r_at_0"] = float(y[np.argmin(x)])
    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "figure": "mehregan_fig3c",
            "method": "pil_trace",
            "source": str(args.image.name),
            "axes": {"x": list(X_RANGE), "y": list(Y_RANGE)},
            "fit": result,
            "series": {"reward": {"xy": {"x": x.tolist(), "y": y.tolist()}}},
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
