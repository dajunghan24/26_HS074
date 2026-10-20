# -*- coding: utf-8 -*-
"""align_measure.py — 정합 오차를 mm 로 잰다.

`align_view.py` 는 "겹쳐 보인다/안 보인다"만 알려준다. 눈으로 보는 판단은
반복해도 수렴하지 않아서, **같은 물체를 두 화면에서 클릭해 그 차이를 mm 로**
돌려주는 도구를 따로 만들었다.

격자는 0.5 mm/px 이므로 픽셀 차이에 0.5 를 곱하면 곧 mm 다.

  1. 뜨겁고 납작한 물체를 조사야 안 아무 데나 둔다 (보정 평면 = 테이블 위)
  2. 왼쪽 thermal 에서 그 물체 중심을 클릭
  3. 오른쪽 optical 에서 같은 물체 중심을 클릭
  4. 오차가 mm 로 뜬다. r 로 다시, q 로 종료

⚠️ 물체가 테이블에서 뜨면 시차 때문에 오차가 커진다. 납작한 것을 평면에 둘 것.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field
import lepton
import optical as optical_mod

CAL = os.path.expanduser("~/Desktop/MLC_row1/mlc_field_calib.json")
# ⚠️ 창 이름은 **ASCII 만** 쓸 것. 한글이 들어가면 cv2 Qt 백엔드가 창을
#    이름으로 못 찾아 setMouseCallback 이 "NULL window handler" 로 죽는다.
WIN = "align measure  (click same spot: thermal then optical   r=reset  q=quit)"

pts = {}          # 0 = thermal, 1 = optical


def on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    gw = param["gw"]
    side = 0 if x < gw else 1
    gx = x if side == 0 else x - gw
    pts[side] = (gx, y)


def main():
    cal = json.load(open(CAL))
    print("보정 저장 시각: %s" % cal.get("saved"))
    tm = field.from_points(cal["cameras"]["thermal"])
    om = field.from_points(cal["cameras"]["optical"])

    cam = lepton.LeptonCapture("/dev/mlc_thermal")
    opt = optical_mod.OpticalCapture(size=(640, 480))
    gh, gw = field.grid_shape()
    mmpx = cal.get("mm_per_px", 0.5)
    print("격자 %d x %d px, %.2f mm/px" % (gw, gh, mmpx))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse, {"gw": gw})

    while True:
        t = cam.read_celsius()
        o = opt.read_bgr()
        if t is None or o is None:
            cv2.waitKey(30)
            continue

        tg = tm.warp(t)
        og = om.warp(o)
        tv = field.colorize(tg)
        ov = og if og.ndim == 3 else cv2.cvtColor(og, cv2.COLOR_GRAY2BGR)
        view = np.hstack([tv, ov])

        for c in range(1, 4):                      # 리프 4열 경계
            x = (c * gw) // 4
            cv2.line(view, (x, 0), (x, gh - 1), (255, 255, 255), 1)
            cv2.line(view, (gw + x, 0), (gw + x, gh - 1), (255, 255, 255), 1)
        cv2.line(view, (0, gh // 2), (2 * gw - 1, gh // 2), (0, 255, 255), 1)

        for side, colour, label in ((0, (0, 255, 0), "T"), (1, (0, 200, 255), "O")):
            if side in pts:
                px, py = pts[side]
                ox = 0 if side == 0 else gw
                cv2.drawMarker(view, (ox + px, py), colour,
                               cv2.MARKER_CROSS, 18, 2)
                cv2.putText(view, label, (ox + px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        if 0 in pts and 1 in pts:
            dx = (pts[1][0] - pts[0][0]) * mmpx
            dy = (pts[1][1] - pts[0][1]) * mmpx
            d = (dx * dx + dy * dy) ** 0.5
            msg = "offset  dx=%+.1f mm  dy=%+.1f mm  |  %.1f mm" % (dx, dy, d)
            colour = (0, 255, 0) if d <= 5.0 else (0, 165, 255) if d <= 15.0 else (0, 0, 255)
            cv2.rectangle(view, (0, gh - 26), (2 * gw - 1, gh - 1), (0, 0, 0), -1)
            cv2.putText(view, msg, (8, gh - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imshow(WIN, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            pts.clear()
        if k == ord("s"):
            cv2.imwrite("/tmp/align_measure.png", view)
            print("[저장] /tmp/align_measure.png")

    cam.release()
    opt.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
