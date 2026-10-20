# -*- coding: utf-8 -*-
"""raw_view.py — 열화상 센서가 주는 그대로를 본다. 아무 처리도 하지 않는다.

코어(mlc_core.py)가 화면에 보내는 그림은 여러 단계를 거칩니다:

    raw -> 회전 -> 조사야 격자로 펴기(4점 보정) -> 기준 프레임 빼기 -> 색칠

그래서 잔상이 보일 때 **센서 문제인지 처리 과정 문제인지** 가릴 수가 없습니다.
이 도구는 그 단계를 전부 걷어내고 **센서 원본만** 띄웁니다. 색칠만 합니다
(온도 배열을 눈으로 볼 수는 없으니 그건 어쩔 수 없습니다).

여기서도 잔상이 보이면 센서 쪽 문제이고, 여기서 깨끗하면 처리 과정 문제입니다.

⚠️ 열화상 장치는 한 프로그램만 열 수 있습니다. 이걸 쓰려면 코어를 먼저 멈춰야 합니다.

  python3 raw_view.py                    # VNC 화면에 창을 띄운다
  python3 raw_view.py --headless --shot /tmp/raw.png   # 한 장만 저장

조작키
  q      종료
  c      FFC 한 번 (셔터를 닫아 기준을 다시 잡는다)
  s      지금 화면을 저장
  space  색 범위 고정/해제 — 고정하면 화면이 흐르는지 눈으로 보기 쉽다
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import ffc
import lepton


def colorize(temp, lo, hi):
    g = np.clip((temp - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)


def annotate(img, temp, lo, hi, frozen):
    lines = [
        "min %.1f   med %.1f   max %.1f C   (span %.1f)"
        % (temp.min(), np.median(temp), temp.max(), temp.max() - temp.min()),
        "color range %.1f ~ %.1f C  %s" % (lo, hi, "[FROZEN]" if frozen else "[auto]"),
        "q quit   c FFC   s save   space freeze",
    ]
    for i, s in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermal", default="/dev/mlc_thermal")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--shot", default="/tmp/raw.png")
    ap.add_argument("--scale", type=int, default=4, help="화면 확대 배수")
    ap.add_argument("--ffc", action="store_true", help="시작할 때 FFC 를 한 번 건다")
    args = ap.parse_args()

    if args.ffc:
        node = ffc.device_node(args.thermal)
        if node and ffc.ensure_mapped(node):
            print("[FFC]", ffc.run_once(node))
            time.sleep(1.5)

    th = lepton.open_thermal(args.thermal, allow_agc=False, force_agc=False)
    temp = None
    t0 = time.time()
    while temp is None and time.time() - t0 < 12:
        temp = th.read_celsius()
        time.sleep(0.02)
    if temp is None:
        print("!! 프레임을 못 받았습니다.")
        return 1

    print("원본 크기: %d x %d" % (temp.shape[1], temp.shape[0]))

    lo, hi = float(np.percentile(temp, 2)), float(np.percentile(temp, 98))
    frozen = False

    if args.headless:
        img = colorize(temp, lo, hi)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(args.shot, annotate(img, temp, lo, hi, frozen))
        print("저장: %s" % args.shot)
        return 0

    win = "RAW thermal (no rotation, no warp, no reference)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, temp.shape[1] * args.scale, temp.shape[0] * args.scale)
    cv2.moveWindow(win, 0, 0)

    while True:
        t = th.read_celsius()
        if t is not None:
            temp = t
        if not frozen:
            lo = float(np.percentile(temp, 2))
            hi = float(np.percentile(temp, 98))

        img = colorize(temp, lo, hi)
        img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                         interpolation=cv2.INTER_NEAREST)
        cv2.imshow(win, annotate(img, temp, lo, hi, frozen))

        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            break
        elif k == ord("c"):
            node = ffc.device_node(args.thermal)
            print("[FFC]", ffc.run_once(node) if node else "장치를 못 찾음")
        elif k == ord("s"):
            cv2.imwrite(args.shot, annotate(img, temp, lo, hi, frozen))
            print("저장: %s" % args.shot)
        elif k == ord(" "):
            frozen = not frozen
            print("색 범위 %s (%.1f ~ %.1f C)"
                  % ("고정" if frozen else "자동", lo, hi))

    cv2.destroyAllWindows()
    try:
        th.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
