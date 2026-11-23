"""ALIGN-pdk-sky130 을 복사해서, LDO 레이아웃에 필요한 빠진 설정을 채운다 (원본은 건드리지 않음).

    python layout/prepare_align_pdk.py ~/eda/ALIGN-pdk-sky130/SKY130_PDK layout/work/SKY130_PDK_ldo

왜 필요한가:
  ALIGN-pdk-sky130 의 저항 생성기(res.py)는 FinFET mock PDK 에서 그대로 가져온 코드라
  layers.json 의 "Cap" 항목에 m1Pitch/m1Width/... 값이 있다고 가정한다. SKY130 layers.json 에는
  이 값이 없어서 저항이 들어간 회로는 KeyError: 'm1Pitch' 로 멈춘다.
  여기서는 그 값을 SKY130 배선 층(M1/M2/M3)의 pitch/width 로 채운다.

  또 ALIGN 기생 추출(-x)이 쓰는 배선 저항 UnitR 이 모든 금속층에 50 ohm/um 로 들어 있다
  (li1 수준 값을 그대로 복사한 것). 실제 SKY130 met1~met5 는 이보다 100배 이상 작으므로,
  SkyWater PDK 문서의 면저항(ohm/sq)과 비아 저항으로 층별 값을 다시 넣는다.

주의:
  이렇게 만든 저항은 ALIGN 의 "금속 지그재그(serpentine) 자리표시 블록"이다. 면적·배치·배선 연습용이며
  실제 SKY130 저항값(400 ohm 등)을 내는 poly 저항(sky130_fd_pr__res_*_po)이 아니다.
"""

# ALIGN 층 이름 -> (SKY130 층, 면저항 ohm/sq)   [SkyWater SKY130 PDK 문서, typical]
SHEET_R = {"M1": ("li1", 12.8), "M2": ("met1", 0.125), "M3": ("met2", 0.125),
           "M4": ("met3", 0.047), "M5": ("met4", 0.047), "M6": ("met5", 0.0285)}
# ALIGN 비아 -> (SKY130 비아, 저항 ohm/개)
VIA_R = {"V0": ("licon", 15.0), "V1": ("mcon", 9.3), "V2": ("via", 4.5),
         "V3": ("via2", 3.41), "V4": ("via3", 3.41), "V5": ("via4", 0.38)}
import json
import shutil
import sys
from pathlib import Path


def main(src, dst):
    src, dst = Path(src).expanduser(), Path(dst).expanduser()
    if not (src / "layers.json").exists():
        raise SystemExit(f"{src}/layers.json 이 없습니다. ALIGN-pdk-sky130/SKY130_PDK 경로를 주세요.")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

    layers = json.loads((dst / "layers.json").read_text())
    by_name = {l["Layer"]: l for l in layers["Abstraction"]}
    cap = by_name["Cap"]
    added = {}
    for i, metal in ((1, "M1"), (2, "M2"), (3, "M3")):
        for key, field in ((f"m{i}Pitch", "Pitch"), (f"m{i}Width", "Width")):
            if key not in cap:
                cap[key] = by_name[metal][field]
                added[key] = cap[key]
    # 기생 추출용 배선 저항: ohm/um = 면저항 / 선폭(um)
    rc = []
    for ly, (name, rs) in SHEET_R.items():
        width_um = by_name[ly]["Width"] / 1000
        r_per_um = round(rs / width_um, 4)
        for k in ("L_3Sigma", "Mean", "U_3Sigma"):
            by_name[ly]["UnitR"][k] = r_per_um
        rc.append(f"{ly}({name}) {r_per_um} ohm/um")
    for ly, (name, r) in VIA_R.items():
        for k in ("L_3Sigma", "Mean", "U_3Sigma"):
            by_name[ly]["R"][k] = r
        rc.append(f"{ly}({name}) {r} ohm")
    (dst / "layers.json").write_text(json.dumps(layers, indent=2))
    print(f"PDK 복사: {src} -> {dst}")
    print(f"Cap 항목에 추가한 값: {added if added else '(없음, 이미 있음)'}")
    print("기생 추출용 저항: " + ", ".join(rc))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
