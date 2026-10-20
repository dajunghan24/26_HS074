# -*- coding: utf-8 -*-
"""diag_map.py — 광카메라 -> 조사야 격자 변환이 멀쩡한지 본다. 읽기 전용."""

import numpy as np
import field
import leaves

SHAPE = (480, 640)

print("조사야 %.0f x %.0f mm, 격자 %s px, %.2f mm/px"
      % (field.APERTURE_W_MM, field.APERTURE_H_MM,
         field.grid_shape(), field.MM_PER_PX))

calib = field.load_calib()
print("보정 파일:", "있음" if calib else "없음")
if calib:
    for k in sorted(calib):
        print("   %-12s %s" % (k, calib[k]))

om = field.map_for("optical", calib, SHAPE,
                   field.OPTICAL_HFOV_DEG, field.OPTICAL_VFOV_DEG)
print("변환 출처:", om.source)

# 1) 화면 전체가 사람이라고 치면 격자도 전부 차야 한다.
full = np.full(SHAPE, 255, np.uint8)
g = om.warp(full)
print("전체 흰 마스크 -> 격자 채움 %.1f%%  (격자 %s)"
      % (100.0 * np.count_nonzero(g) / g.size, g.shape))

# 2) 왼쪽 60% 만 사람이라고 치면?
half = np.zeros(SHAPE, np.uint8)
half[:, :int(SHAPE[1] * 0.6)] = 255
g2 = (om.warp(half) > 127).astype(np.uint8) * 255
print("왼쪽 60%% 마스크 -> 격자 채움 %.1f%%"
      % (100.0 * np.count_nonzero(g2) / g2.size))

gh, gw = field.grid_shape()
prof = leaves.top_profile(g2)
raw, infos = leaves.openings_from_profile(prof, g2, gh, gw, field.MM_PER_PX,
                                          n_cols=4, margin_mm=5.0)
print("   -> open_mm", [round(v, 1) for v in raw])
print("   -> infos  ", [(i["y_top"], i["pix"]) for i in infos])

# 3) 격자 네 귀퉁이가 광카메라 화면 어디로 가는가
import cv2
M = getattr(om, "M", None)
if M is not None:
    c = np.float32([[0, 0], [gw - 1, 0], [gw - 1, gh - 1], [0, gh - 1]]
                   ).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(c, np.linalg.inv(M)).reshape(-1, 2)
    print("조사야 네 귀퉁이의 광카메라 좌표 (화면은 640x480):")
    for name, p in zip(["좌상", "우상", "우하", "좌하"], pts):
        print("   %s  (%7.1f, %7.1f)" % (name, p[0], p[1]))
