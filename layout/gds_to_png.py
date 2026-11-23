"""GDS 레이아웃을 PNG 그림으로 그린다 (KLayout 같은 GUI 없이 결과를 빠르게 보기용).

    python layout/gds_to_png.py layout/work/ldo_rl/run/LDO_RL_0.gds -o layout/ldo_rl.png

pip install gdstk matplotlib
"""
import argparse
from collections import defaultdict

import gdstk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

# SKY130 GDS 층 번호 -> (이름, 색).  표에 없는 층은 회색으로 그린다.
SKY130 = {
    (64, 20): ("nwell", "#c9b3e6"), (65, 20): ("diff", "#7fbf7f"), (65, 44): ("tap", "#5fa05f"), (66, 20): ("poly", "#e06666"),
    (66, 44): ("licon", "#444444"), (67, 20): ("li1", "#9fc5e8"), (67, 44): ("mcon", "#333333"),
    (68, 20): ("met1", "#6fa8dc"), (68, 44): ("via", "#222222"), (69, 20): ("met2", "#f6b26b"),
    (69, 44): ("via2", "#222222"), (70, 20): ("met3", "#93c47d"), (70, 44): ("via3", "#222222"),
    (71, 20): ("met4", "#c27ba0"), (71, 44): ("via4", "#222222"), (72, 20): ("met5", "#ffd966"),
    (89, 44): ("capm", "#76a5af"), (93, 44): ("nsdm", "#ffe599"), (94, 20): ("psdm", "#d5a6bd"),
}
# 넓게 깔리는 층은 아주 옅게 (아래 층이 가려지지 않게)
FAINT = {(64, 20), (93, 44), (94, 20)}
# 아래에서 위 순서로 그리기 (위 층이 위에 보이도록)
ORDER = [(64, 20), (93, 44), (94, 20), (65, 20), (65, 44), (66, 20), (66, 44), (67, 20), (67, 44),
         (68, 20), (68, 44), (69, 20), (69, 44), (70, 20), (70, 44), (71, 20), (71, 44), (72, 20), (89, 44)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gds")
    ap.add_argument("-o", "--out", default="layout.png")
    ap.add_argument("--cell", help="그릴 셀 이름 (기본: 최상위 셀)")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    lib = gdstk.read_gds(a.gds, unit=1e-6)   # 좌표를 um 단위로 읽는다 (ALIGN GDS 는 user unit 이 1m)
    top = next(c for c in lib.cells if c.name == a.cell) if a.cell else lib.top_level()[0]
    polys = defaultdict(list)
    for p in top.get_polygons():          # 계층을 펼쳐서 모든 도형을 가져온다
        polys[(p.layer, p.datatype)].append(p.points)
    (x0, y0), (x1, y1) = top.bounding_box()
    w, h = x1 - x0, y1 - y0

    fig, ax = plt.subplots(figsize=(10, 10 * h / w if w else 10))
    keys = [k for k in ORDER if k in polys] + sorted(k for k in polys if k not in ORDER)
    handles = []
    for k in keys:
        if k not in SKY130:   # 경계·표시용 층(셀 외곽선 등)은 테두리만
            ax.add_collection(PolyCollection(polys[k], facecolor="none", edgecolor="#999999", linewidth=0.3))
            continue
        name, color = SKY130[k]
        alpha = 0.12 if k in FAINT else (0.9 if k[1] == 44 else 0.55)
        ax.add_collection(PolyCollection(polys[k], facecolor=color, edgecolor="none", alpha=alpha))
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=max(alpha, 0.4), label=f"{name} ({len(polys[k])})"))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"{top.name}  {w:.1f} x {h:.1f} um  ({w * h:,.0f} um^2)")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(a.out, dpi=a.dpi, bbox_inches="tight")
    print(f"{a.out}: {top.name} {w:.1f} x {h:.1f} um, {sum(len(v) for v in polys.values()):,} polygons")


if __name__ == "__main__":
    main()
