#!/usr/bin/env bash
# RL 이 찾은 LDO 크기 -> ALIGN 자동 레이아웃(GDS) -> PNG 그림
#
#   bash layout/run_ldo_layout.sh                                  # 기본 예제 설계
#   bash layout/run_ldo_layout.sh --log train.log --task 25        # 학습 로그에서 성공한 설계 골라서
#   bash layout/run_ldo_layout.sh --params "4.819 1.789 ... 4.08e-10"
#   bash layout/run_ldo_layout.sh --with_rc                        # 보상 R/C 까지 ALIGN 으로 (오류 있음, 9-7 참고)
#
# 필요한 것 (docs/MANUAL.md 9장): conda 환경 align (ALIGN 설치됨), ALIGN-pdk-sky130 클론
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ALIGN_ENV="${ALIGN_ENV:-$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/envs/align}"   # ALIGN 을 설치한 conda 환경
ALIGN_PDK="${ALIGN_PDK:-$HOME/eda/ALIGN-pdk-sky130/SKY130_PDK}"                              # SKY130 용 ALIGN PDK
WORK="${WORK:-$REPO/layout/work}"
NAME="${NAME:-ldo_rl}"
PY="$ALIGN_ENV/bin/python"

[ -x "$PY" ] || { echo "ALIGN 환경을 찾지 못했습니다: $ALIGN_ENV  (ALIGN_ENV=... 로 지정)"; exit 1; }
[ -f "$ALIGN_PDK/layers.json" ] || { echo "ALIGN SKY130 PDK 를 찾지 못했습니다: $ALIGN_PDK  (ALIGN_PDK=... 로 지정)"; exit 1; }

echo "== 1/4 PDK 준비 (원본을 복사해 빠진 저항 설정을 채움, --with_rc 일 때 필요)"
"$PY" "$REPO/layout/prepare_align_pdk.py" "$ALIGN_PDK" "$WORK/SKY130_PDK_ldo"

echo "== 2/4 RL 크기 -> ALIGN 넷리스트"
"$PY" "$REPO/layout/ldo_to_align.py" --out "$WORK/$NAME" "$@"

echo "== 3/4 ALIGN 자동 배치·배선 (수 분 걸립니다)"
rm -rf "$WORK/$NAME/run" && mkdir -p "$WORK/$NAME/run"
cd "$WORK/$NAME/run"
"$PY" "$ALIGN_ENV/bin/schematic2layout.py" .. -p "$WORK/SKY130_PDK_ldo" -n 1 -x > align.log 2>&1 || true   # -x: 배선 기생 R/C 추출 (.cir)
sed 's/\x1b\[[0-9;]*m//g' align.log | grep -E "ERROR" || true
if grep -q "LVS / DRC errors found" align.log; then
    echo "[주의] ALIGN 자체 LVS/DRC 검사에서 오류가 있습니다 (위 ERROR 줄 참고)"
else
    echo "ALIGN 자체 LVS/DRC 검사: 오류 없음"
fi
GDS="$(ls "$WORK/$NAME/run/"*.gds 2>/dev/null | grep -v "\.python\.gds" | head -1 || true)"
[ -n "$GDS" ] || { echo "GDS 가 만들어지지 않았습니다. $WORK/$NAME/run/align.log 를 확인하세요."; exit 1; }

echo "== 4/4 GDS -> PNG"
"$PY" "$REPO/layout/gds_to_png.py" "$GDS" -o "$WORK/$NAME/$NAME.png"
echo "완료: GDS = $GDS"
echo "      PNG = $WORK/$NAME/$NAME.png"
