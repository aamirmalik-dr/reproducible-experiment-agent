from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    "gain_by_arm.png",
    "per_task_gain.png",
    "trial_trajectories.png",
    "dev_vs_holdout.png",
    "trial_outcomes.png",
]


def _draw(out: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--out", str(out)],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def test_make_figures_writes_every_png_from_saved_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    first, second = tmp_path / "a", tmp_path / "b"
    _draw(first)
    _draw(second)
    for name in EXPECTED:
        png = first / name
        assert png.exists() and png.stat().st_size > 10_000, name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert png.stat().st_size < 400_000, f"{name} is over 400 KB"
        assert png.read_bytes() == (second / name).read_bytes(), f"{name} is not reproducible"
        assert (ROOT / "docs" / "figures" / name).exists(), f"{name} is not committed"
