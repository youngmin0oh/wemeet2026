"""ALIGN 격자(fin) 반올림: 기본 예제 설계는 그대로, 큰 소자는 X x Y 로 나눌 수 있는 단위 셀 수."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "layout"))

from ldo_to_align import DEFAULT, build, elongation, quantize_w  # noqa: E402


def test_default_design_keeps_its_layout_sizes():
    _, _, _, eff = build(DEFAULT)
    assert round(eff["M1"][0], 3) == 5.04
    assert round(eff["M3"][0], 3) == 68.04
    assert round(eff["M5"][0], 3) == 30.24
    assert round(eff["M6"][0], 3) == 7148.4      # m = 1702 = 23 x 74


def test_large_device_avoids_strip_only_unit_counts():
    assert elongation(1514) > 300               # 2 x 757: only strips
    nfin, nf, m = quantize_w(6357.7)
    assert (nfin, nf, m) == (10, 2, 1512)       # 36 x 42
    assert elongation(m) < 1.2
    assert abs(0.21 * nfin * nf * m - 6357.7) / 6357.7 < 0.02


def test_quantized_width_stays_inside_the_model_range():
    nfin, nf, m = quantize_w(100.0)               # 10 x 2 x 24 = 100.8 um would leave the model
    assert 0.21 * nfin * nf * m <= 100.0
    assert abs(0.21 * nfin * nf * m - 100.0) / 100.0 <= 0.02
    nfin, nf, m = quantize_w(100.0 * 150, w_max=100.0 * 150)
    assert 0.21 * nfin * nf * m <= 100.0 * 150
