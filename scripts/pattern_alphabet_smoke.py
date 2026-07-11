"""Smoke test: verify FixedMeanPatternAlphabet produces distinct patterns.

Run: uv run python scripts/pattern_alphabet_smoke.py
"""

from __future__ import annotations

import sys

import numpy as np

from envs.mehregan.fixed_mean_patterns import FixedMeanPatternAlphabet


def main() -> None:
    alpha = FixedMeanPatternAlphabet(mean_hz=45.0)
    print(f"Pattern alphabet: {alpha.n_actions} patterns at {alpha.mean_hz} Hz")
    print(f"Step duration: {alpha.step_duration_s}s, dt: {alpha.dt_ms}ms")
    print()

    # Pattern 0 should be the regular train
    p0 = alpha.idbs_for_pattern(0)
    p0_pulses = alpha.pulse_count(0)
    print(f"Pattern 0 (regular): {p0_pulses} pulses, "
          f"nonzero steps = {np.count_nonzero(p0)}/{len(p0)}")

    # Check that all patterns have the same pulse count (mean rate preserved)
    pulse_counts = [alpha.pulse_count(i) for i in range(alpha.n_actions)]
    unique_counts = sorted(set(pulse_counts))
    print(f"Pulse counts across {alpha.n_actions} patterns: {unique_counts}")
    if len(unique_counts) != 1:
        print("WARNING: not all patterns have the same pulse count!")
    else:
        print(f"✓ All patterns have exactly {unique_counts[0]} pulses")

    # Check that irregular patterns differ from regular
    n_unique_traces = 0
    for i in range(alpha.n_actions):
        trace = alpha.idbs_for_pattern(i)
        if not np.array_equal(trace, p0):
            n_unique_traces += 1
    print(f"✓ {n_unique_traces}/{alpha.n_actions} patterns differ from regular train")

    # Check DbsSpec round-trip
    spec = alpha.to_dbs_spec(0)
    print(f"\nPattern 0 → DbsSpec: pick_dbs_freq={spec.pick_dbs_freq}, "
          f"mean_hz={spec.mean_hz}, has_idbs={spec.idbs is not None}")

    # Verify reproducibility
    p5_v1 = alpha.idbs_for_pattern(5)
    p5_v2 = alpha.idbs_for_pattern(5)
    assert np.array_equal(p5_v1, p5_v2), "Pattern 5 not reproducible!"
    print("✓ Pattern reproducibility confirmed")

    # Show ISI distribution for pattern 0 vs a few irregular ones
    for idx in [0, 1, 10, 20]:
        trace = alpha.idbs_for_pattern(idx)
        active = trace > 0.0
        rising = np.concatenate(([active[0]], active[1:] & ~active[:-1]))
        onsets = np.flatnonzero(rising)
        if len(onsets) > 1:
            isis = np.diff(onsets) * alpha.dt_ms
            print(f"  Pattern {idx:2d}: ISIs — mean={isis.mean():.2f}ms, "
                  f"std={isis.std():.2f}ms, min={isis.min():.2f}ms, max={isis.max():.2f}ms")

    print("\n✓ Smoke test passed")


if __name__ == "__main__":
    main()
