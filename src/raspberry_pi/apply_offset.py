# -*- coding: utf-8 -*-
"""apply_offset.py — 잰 정합 오차만큼 열화상 보정점을 옮긴다.

`align_measure.py` 가 돌려주는 (dx, dy) 는 **광학 − 열화상** 이다. 즉 열화상이
그만큼 반대쪽으로 밀려 보인다는 뜻이다. 네 점을 다시 클릭하는 대신, 그 값으로
**열화상 네 점만** 옮겨 두 카메라를 맞춘다.

  python3 apply_offset.py <dx_mm> <dy_mm>

원리: 지금 호모그래피 H 는 열화상 네 점 → 격자 네 귀퉁이 다. 새 네 점을
`H^-1(귀퉁이 − (dx,dy)/mm_per_px)` 로 잡으면, 열화상 내용이 격자에서 정확히
(dx, dy) 만큼 광학 쪽으로 이동한다.

⚠️ 광학 보정점은 건드리지 않는다. 기준은 광학이다 (화각이 좁아 더 엄격하다).
⚠️ 실행 전 자동으로 백업한다. 되돌리려면 그 파일을 복사하면 된다.
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field

CAL = os.path.expanduser("~/Desktop/MLC_row1/mlc_field_calib.json")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    dx_mm, dy_mm = float(sys.argv[1]), float(sys.argv[2])

    cal = json.load(open(CAL))
    mmpx = cal.get("mm_per_px", 0.5)
    src = np.float32(cal["cameras"]["thermal"])
    gh, gw = field.grid_shape()
    dst = np.float32([[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]])

    H = cv2.getPerspectiveTransform(src, dst)
    Hinv = np.linalg.inv(H)

    off = np.float32([dx_mm / mmpx, dy_mm / mmpx])
    shifted = (dst - off).reshape(-1, 1, 2)
    new_src = cv2.perspectiveTransform(shifted, Hinv).reshape(-1, 2)

    print("격자 %d x %d px, %.2f mm/px" % (gw, gh, mmpx))
    print("적용할 오차: dx=%+.1f mm  dy=%+.1f mm  (= %+.1f, %+.1f px)"
          % (dx_mm, dy_mm, off[0], off[1]))
    print()
    print("%-6s %-22s %-22s %s" % ("", "이전 (px)", "이후 (px)", "이동"))
    names = ["좌상", "우상", "우하", "좌하"]
    for i, n in enumerate(names):
        a, b = src[i], new_src[i]
        print("%-6s (%6.1f, %6.1f)      (%6.1f, %6.1f)      (%+.1f, %+.1f)"
              % (n, a[0], a[1], b[0], b[1], b[0] - a[0], b[1] - a[1]))

    h, w = 120, 160          # Lepton 센서
    out = [p for p in new_src.tolist()]
    bad = [i for i, p in enumerate(out)
           if not (0 <= p[0] < w and 0 <= p[1] < h)]
    if bad:
        print()
        print("⚠️ 센서 밖으로 나간 점이 있습니다: %s" % [names[i] for i in bad])
        print("   오차가 너무 크거나 부호가 반대입니다. 적용하지 않았습니다.")
        return 1

    bak = CAL + ".bak-" + time.strftime("%m%d_%H%M%S") + "-offset"
    shutil.copyfile(CAL, bak)
    cal["cameras"]["thermal"] = [[round(float(x), 2) for x in p] for p in out]
    cal["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cal["note"] = ("조사야 네 귀퉁이 — 열화상은 정합 오차 "
                   "dx=%+.1f dy=%+.1f mm 를 반영해 보정함" % (dx_mm, dy_mm))
    with open(CAL, "w", encoding="utf-8") as f:
        json.dump(cal, f, ensure_ascii=False, indent=2)
    print()
    print("[OK] 적용 완료.  백업: %s" % os.path.basename(bak))
    return 0


if __name__ == "__main__":
    sys.exit(main())
