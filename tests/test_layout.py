"""The packages import, and every paper has a pawtrace spec with its rights set."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_packages_import():
    import adapters  # noqa: F401
    import controllers.baselines  # noqa: F401
    import controllers.rl.ddpg  # noqa: F401
    import controllers.rl.sea_dbs  # noqa: F401
    import controllers.rl.snn  # noqa: F401
    import plants.parkinsons.kumaravelu2016  # noqa: F401


def test_ieee_copies_never_go_to_ai_tools():
    # Source rules (AGENTS.md): Mehregan 2024 and Nguyen ISCAS 2025 are human intake only.
    for paper in ("mehregan", "nguyen"):
        spec = yaml.safe_load((ROOT / "papers" / paper / "spec.yaml").read_text())
        assert spec["rights"]["rules"]["ai_tools"] == "never", paper


def test_off_limits_sources_listed_everywhere():
    must = {"rl-adaptive-dbs-legacy", "kumaravelu-plant", "modeldb.science/206232",
            "modeldbrepository/206232", "fleming2020-code"}
    for spec_path in sorted((ROOT / "papers").glob("*/spec.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        assert must <= set(spec["policy"]["off_limits"]), spec_path


def test_ravivarapu_uses_the_arxiv_copy():
    spec = yaml.safe_load((ROOT / "papers" / "ravivarapu" / "spec.yaml").read_text())
    assert "arxiv.org/abs/2507.06326" in spec["rights"]["copy_url"]
