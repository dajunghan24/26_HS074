#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""조사야 사진에서 **열별로** 열린 높이를 잰다 (px -> mm).

## 왜 열별인가

리프 4장이 각자 다른 위치에 서면 조사야 아래·위 경계가 **계단 모양**이 된다.
사각형 하나로 맞추면 그 계단이 평균으로 뭉개져 채널별 차이를 못 본다.
4·8번 날이 5mm 긴 것도 열별로 봐야 드러난다.

## 자는 조사야 가로폭

우리 리프는 **위아래로만** 움직이므로 가로폭은 어떤 자세에서도 변하지 않는다.
그래서 가로폭을 자로 삼으면 확대율(선원-리프-테이블 거리)을 몰라도 된다.
`--width-mm` 으로 그 실제 폭을 준다 (기본 95mm = MLC 개구부 가로).

## 하는 일

1. 밝은 영역을 찾아 기울기를 잰다 (문턱값을 여러 개 시도해 밝은 방에서도 됨)
2. 그 기울기만큼 사진을 되돌려 조사야를 화면 축에 맞춘다
3. 열마다 위·아래 경계를 찾아 높이를 px 로 잰다
4. 가로폭을 자로 mm 로 환산한다
"""
import argparse
import cv2
import numpy as np


def find_field(gray):
    """밝은 조사야의 minAreaRect 를 찾는다. 못 찾으면 None."""
    h, w = gray.shape
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    best = None
    for pct in range(60, 100):
        t = float(np.percentile(gray, pct))
        if t >= gray.max():
            continue
        _, m = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if not (0.02 <= a / float(w * h) <= 0.70):
                continue
            r = cv2.minAreaRect(c)
            box = cv2.boxPoints(r)
            if (box[:, 0].min() < 3 or box[:, 0].max() > w - 3
                    or box[:, 1].min() < 3 or box[:, 1].max() > h - 3):
                continue          # 잘렸으면 각도를 못 믿는다
            rw, rh = r[1]
            if rw < 40 or rh < 40 or a / (rw * rh) < 0.80:
                continue
            if best is None or a > best[0]:
                best = (a, r, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--width-mm", type=float, default=95.0,
                    help="조사야 가로 실제 폭 [mm] (기본 95 = MLC 개구부)")
    ap.add_argument("--cols", type=int, default=4, help="리프 열 개수")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        print("사진을 못 읽었습니다:", a.image)
        return 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = find_field(gray)
    if found is None:
        print("조사야를 못 찾았습니다. 방 조명을 끄고 램프만 켜 보세요.")
        return 2
    area, rect, thr = found
    (cx, cy), (rw, rh), ang = rect
    angle = ang if rw >= rh else ang + 90.0
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90

    # 기울기만큼 되돌려 조사야를 화면 축에 맞춘다.
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
    # ⚠️ 문턱값을 **50% 로 다시 잡는다.** 조사야 가장자리는 반음영이라 흐릿해서,
    # 탐색용 문턱값(밝기 상위 몇 %)을 그대로 쓰면 사진 밝기에 따라 경계가 밀린다.
    # 실제로 같은 자세인데 가로폭이 ±8% 흔들렸다. 조사야 측정의 표준대로
    # **밝은 영역 대표값과 배경 대표값의 중간**을 경계로 삼으면 안정된다.
    _, rough = cv2.threshold(rot, thr, 255, cv2.THRESH_BINARY)
    plateau = float(np.median(rot[rough > 0])) if np.any(rough) else float(rot.max())
    background = float(np.median(rot[rough == 0]))
    half = 0.5 * (plateau + background)
    _, mask = cv2.threshold(rot, half, 255, cv2.THRESH_BINARY)
    print("문턱값    : 배경 %.0f / 조사야 %.0f -> 50%% 경계 %.0f" % (background, plateau, half))

    # ⚠️ 십자선(중심선)이 조사야를 네 조각으로 쪼갠다. 메우지 않으면 아래에서
    # "가장 큰 덩어리"가 조사야가 아니라 그 **반쪽**이 되어 가로폭이 절반으로
    # 나온다. 2026-09-07 에 실제로 309px(실제 670px)이 나왔다.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41)))

    # ⚠️ 밝은 화소를 전부 세면 안 된다. 조사야 밖의 밝은 물체(반사, 라벨)까지
    # 폭에 들어가 가로폭이 두 배로 나온다. **가장 큰 덩어리만** 남긴다.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("되돌린 뒤 밝은 덩어리가 없습니다.")
        return 2
    big = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [big], -1, 255, cv2.FILLED)

    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        print("되돌린 뒤 밝은 화소가 너무 적습니다.")
        return 2
    x0, x1 = int(xs.min()), int(xs.max())
    W_px = x1 - x0 + 1
    mm_px = a.width_mm / float(W_px)

    print("사진      : %s" % a.image)
    print("조사야 기울기: %+.2f 도  (되돌려서 잼)" % angle)
    print("가로폭    : %d px  = %.1f mm  ->  %.4f mm/px" % (W_px, a.width_mm, mm_px))
    print()

    # 열마다 위·아래 경계. 열 경계 근처는 리프 사이 틈이라 안쪽 60%만 본다.
    print(" 열   열린높이(px)   열린높이(mm)")
    heights = []
    for i in range(a.cols):
        cl = x0 + W_px * i / a.cols
        cr = x0 + W_px * (i + 1) / a.cols
        pad = (cr - cl) * 0.20
        lo, hi = int(cl + pad), int(cr - pad)
        hs = []
        for x in range(lo, hi + 1):
            col = np.nonzero(mask[:, x])[0]
            if len(col) > 5:
                hs.append(col.max() - col.min() + 1)
        if not hs:
            print("  %d      (측정 실패)" % (i + 1))
            heights.append(None)
            continue
        hp = float(np.median(hs))
        heights.append(hp * mm_px)
        print("  %d      %6.1f        %6.2f" % (i + 1, hp, hp * mm_px))

    good = [v for v in heights if v is not None]
    if len(good) >= 2:
        print()
        print("열 사이 최대 차이: %.2f mm" % (max(good) - min(good)))
        print("평균 열린 높이  : %.2f mm" % (sum(good) / len(good)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
