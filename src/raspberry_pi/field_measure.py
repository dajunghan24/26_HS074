#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""광카메라 사진에서 조사야(밝은 사각형)를 찾아 회전각과 조준 어긋남을 잰다.

## 밝은 방에서도 되게 만든 이유

Otsu 한 방으로 가르면 **방 조명이 켜져 있을 때 테이블 상판이 조사야만큼 밝아서**
상판 전체가 잡힌다. 2026-09-07 에 두 번 헛짚었다. 증상은 늘 같다:
회전이 딱 0.00도 나오고, 사각형이 화면 네 변에 붙는다.

그래서 문턱값을 하나로 정하지 않고 **여러 개를 시도해 걸러낸다.** 조사야는

  - 화면 안에 온전히 들어와 있고 (네 변에 안 붙는다)
  - 화면의 5~60% 를 차지하며
  - 사각형에 가깝다 (덩어리 넓이 / 최소외접사각형 넓이 > 0.85)

이 세 조건을 통과하는 것 중 **가장 큰 것**을 고른다. 하나도 없으면 실패라고
말한다 — 엉뚱한 숫자를 내놓는 것보다 낫다.
"""
import sys
import cv2
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opt.jpg"
img = cv2.imread(path)
if img is None:
    print("사진을 못 읽었습니다:", path)
    raise SystemExit(1)

h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("사진 %dx%d   화면 중심 (%.0f, %.0f)" % (w, h, w / 2.0, h / 2.0))
print("밝기: 중앙값 %d  최대 %d" % (int(np.median(gray)), gray.max()))

# 십자선이 조사야를 넷으로 쪼개므로 닫기 연산으로 먼저 이어 붙인다.
kern = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))

best = None
tried = []
clipped = []
for pct in range(70, 100):
    t = float(np.percentile(gray, pct))
    if t >= gray.max():
        continue
    _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        frac = area / float(w * h)
        if not (0.05 <= frac <= 0.60):
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        # 화면 변에 붙으면 잘린 것이다 — 각도를 믿을 수 없다.
        m = 3
        if (box[:, 0].min() < m or box[:, 0].max() > w - m
                or box[:, 1].min() < m or box[:, 1].max() > h - m):
            continue
        rw, rh = rect[1]
        if rw < 40 or rh < 40:
            continue
        fill = area / (rw * rh)          # 사각형에 얼마나 가까운가
        if fill < 0.85:
            continue
        tried.append((area, pct, rect, fill))

    # 화면 밖으로 잘린 후보도 넓이만 따로 기억해 둔다. 잘린 조사야를 버리고
    # 그 **사분면 하나**를 조사야로 착각하는 것을 막기 위한 것이다.
    for c in cnts:
        a2 = cv2.contourArea(c)
        if a2 / float(w * h) < 0.05:
            continue
        b2 = cv2.boxPoints(cv2.minAreaRect(c))
        if (b2[:, 0].min() < 3 or b2[:, 0].max() > w - 3
                or b2[:, 1].min() < 3 or b2[:, 1].max() > h - 3):
            clipped.append(a2)

if not tried:
    print()
    print("조사야를 못 찾았습니다.")
    print("  - 방 조명을 끄고 조사야 램프만 켜면 확실합니다")
    print("  - 조사야가 화면 밖으로 잘려도 못 찾습니다 (각도를 못 믿기 때문)")
    raise SystemExit(2)

area, pct, rect, fill = max(tried, key=lambda r: r[0])

# 잘린 덩어리가 고른 것보다 훨씬 크면, 고른 것은 조사야가 아니라 그 일부다.
if clipped and max(clipped) > 1.6 * area:
    print()
    print("조사야가 화면 밖으로 잘렸습니다 — 각도를 믿을 수 없어 멈춥니다.")
    print("  (잘린 덩어리 %.0f px 가 고른 후보 %.0f px 보다 훨씬 큽니다)"
          % (max(clipped), area))
    print("  카메라를 뒤로 빼거나 각도를 돌려 조사야 전체가 들어오게 하세요.")
    raise SystemExit(2)
(cx, cy), (rw, rh), ang = rect
print("고른 문턱값: 상위 %d%%   사각형다움 %.2f" % (100 - pct, fill))

a = ang if rw >= rh else ang + 90.0
while a > 45:
    a -= 90
while a < -45:
    a += 90

print()
print("--- 조사야 ---")
print("  중심   : (%.0f, %.0f)" % (cx, cy))
print("  크기   : %.0f x %.0f px" % (rw, rh))
print("  회전   : %+.2f 도" % a)

dx, dy = cx - w / 2.0, cy - h / 2.0
# 화각 66도(Camera Module 3 수평) 기준 화소당 각도. tan 모델을 쓴다.
f_px = (w / 2.0) / np.tan(np.deg2rad(33.0))
pan = np.rad2deg(np.arctan2(dx, f_px))
tilt = np.rad2deg(np.arctan2(dy, f_px))

print()
print("--- 조준 어긋남 ---")
print("  가로 %+.0f px  -> 카메라를 %s 으로 %.1f 도" %
      (dx, "왼쪽" if dx < 0 else "오른쪽", abs(pan)))
print("  세로 %+.0f px  -> 카메라를 %s 로 %.1f 도" %
      (dy, "아래" if dy > 0 else "위", abs(tilt)))
print("  회전 %+.2f 도 -> 카메라를 %s 로 %.2f 도" %
      (a, "반시계" if a > 0 else "시계", abs(a)))

margin = min(cx - rw / 2.0, w - (cx + rw / 2.0), cy - rh / 2.0, h - (cy + rh / 2.0))
print()
print("가장 좁은 여백: %.0f px" % margin)
if margin < 30:
    print("⚠️ 여백이 좁습니다 — 조금만 틀어져도 조사야가 잘립니다.")

out = img.copy()
cv2.drawContours(out, [cv2.boxPoints(rect).astype(int)], 0, (0, 0, 255), 3)
cv2.circle(out, (int(cx), int(cy)), 9, (0, 0, 255), -1)
cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 44, 3)
cv2.line(out, (int(cx), int(cy)), (w // 2, h // 2), (0, 255, 255), 2)
cv2.imwrite("/tmp/opt_measured.jpg", out)
print("표시한 그림: /tmp/opt_measured.jpg")
