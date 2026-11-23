# layout/ — RL 크기 → ALIGN 자동 레이아웃 → post-layout 확인

자세한 설치·사용법은 [docs/MANUAL.md 9장](../docs/MANUAL.md#9-레이아웃-자동-생성-align) 을 보세요.

| 파일 | 하는 일 | 실행 환경 |
|---|---|---|
| `run_ldo_layout.sh` | 아래 1–4 를 한 번에 (PDK 준비 → 넷리스트 → ALIGN → PNG) | `align` |
| `prepare_align_pdk.py` | ALIGN-pdk-sky130 을 복사해 빠진 저항 설정과 실제 SKY130 배선 저항값을 넣음 | `align` |
| `ldo_to_align.py` | RL 이 찾은 LDO 크기 13개 → ALIGN 넷리스트 (fin/stack 격자로 반올림) | `align` |
| `gds_to_png.py` | GDS → PNG 그림 | `align` |
| `check_ldo_quantized.py` | 반올림한 크기로 9개 PVT 를 다시 시뮬레이션 (pre-layout) | `analog_sizing` |
| `postlayout_ldo.py` | ALIGN 배선 기생 R/C 를 넣고 9개 PVT 를 다시 시뮬레이션 (post-layout) | `analog_sizing` |

```bash
conda activate align         && bash layout/run_ldo_layout.sh
conda activate analog_sizing && python layout/check_ldo_quantized.py && python layout/postlayout_ldo.py
```

결과물은 `layout/work/` 에 생기며 git 에는 올라가지 않습니다.
