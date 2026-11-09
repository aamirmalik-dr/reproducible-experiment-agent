from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEMS = ["fig1_main", "figS1_paired_differences"]
SVG_LIMIT = 600_000
PNG_LIMIT = 500_000


def _draw(out: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_figures.py"), "--out", str(out)],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )


def test_figure_style_uses_the_vendored_typeface() -> None:
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import figure_style

        assert figure_style.apply() == "Liberation Sans"
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_make_figures_writes_every_svg_and_png_from_saved_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    first, second = tmp_path / "a", tmp_path / "b"
    _draw(first)
    _draw(second)
    for stem in STEMS:
        svg, png = first / f"{stem}.svg", first / f"{stem}.png"
        assert svg.exists() and 10_000 < svg.stat().st_size < SVG_LIMIT, svg.name
        assert png.exists() and 10_000 < png.stat().st_size < PNG_LIMIT, png.name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        text = svg.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<?xml") and "<text" not in text, svg.name
        for path in (svg, png):
            assert path.read_bytes() == (second / path.name).read_bytes(), path.name
            assert (ROOT / "docs" / "figures" / path.name).exists(), f"{path.name} not committed"
    assert (first / "figure_stats.json").read_bytes() == (second / "figure_stats.json").read_bytes()
