# -*- coding: utf-8 -*-
"""
field.py — 두 카메라를 '조사야 평면' 하나로 모으는 정합

이 프로젝트에서 가장 중요한 개념입니다.

열화상과 광카메라는 서로 다른 자리에 붙어 있고, 화각도 해상도도 다릅니다.
두 영상을 그냥 겹치면 몇 cm 씩 어긋납니다. 조사야가 95x115mm 밖에 안 되는데
관구에서 10cm 만 떨어져 달아도 리프 평면에서 수십 mm 가 밀립니다.

해결은 '거리를 재서 계산'이 아니라 '네 점을 찍는 것'입니다.
조사야(리프가 다 열렸을 때 뚫리는 사각형) 네 귀퉁이를 각 카메라에서 클릭하면,
호모그래피 한 장이 오프셋/기울기/회전/배율/렌즈 차이를 통째로 흡수합니다.
두 카메라를 각각 같은 사각형에 맞춰 펴면, 결과 영상 두 장은 자동으로 같은
좌표계에 놓입니다. 이게 융합 판정의 전제입니다.

  격자 좌표계:  0.5 mm/px,  95mm x 115mm  ->  190 x 230 px
  격자의 (0,0) = 조사야 좌상단,  가로 = 리프가 늘어선 방향(4열)
"""

import json
import os
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(BASE_DIR, "mlc_field_calib.json")

# 조사야 실측값 (2026-08-25 사용자 실측: DXC-89ML 콜리메이터 빔 개구 95 x 115 mm)
# 조사야 개구부 [mm] — 2026-09-07 실측으로 갱신.
#
# 옛값 95 x 115 는 2026-08-25 에 잰 **X선 콜리메이터의 빔 개구부**였다. 그런데
# 리프는 판 구멍(150x150)을 덮도록 만들어져 실제로는 더 넓은 폭을 담당한다:
#   날 4장 x 38.7mm - 같은 행 겹침 1.5mm x 3 = 150.3mm  (실측 149.508mm 와 일치)
#
# 이 값은 **카메라 화면을 리프 4열에 나눠 배정하는 기준**이다. 실제보다 좁게
# 잡으면 사람의 어느 부위가 몇 번 리프 담당인지가 어긋난다.
#
# ⚠️ 콜리메이터를 이 개구부에 맞춰 열어 둔 상태를 전제로 한다. 빔을 더 좁히면
#    리프가 빈 곳을 담당하게 되므로 이 값도 같이 줄일 것.
APERTURE_W_MM = 149.5
APERTURE_H_MM = 110.0

# 격자 해상도. 0.5mm/px 면 리프 1mm 를 2픽셀로 보므로 충분합니다.
MM_PER_PX = 0.5

# 보정 없이 급하게 돌릴 때 쓰는 기하 추정값 (융합에는 쓰지 마세요)
THERMAL_HFOV_DEG = 57.0     # Lepton 3.5 표준 렌즈
THERMAL_VFOV_DEG = 44.3
# ⚠️ 데이터시트 값(66 x 41)은 **16:9 전체 센서** 기준이다. GUI 는 4:3(640x480)
# 으로 받으므로 가로가 잘린다: 1296*4/3 = 1728 / 2304 = 75%.
#   가로 = 2*atan(tan(33도) * 0.75) = 51.9도,  세로는 그대로 40.1도
# 2026-09-07 에 조사야 램프로 확인했다. 이 값이 실제와 다르면 화면 픽셀을
# 조사야 mm 로 옮기는 계산이 통째로 틀어진다.
# GUI 를 16:9 로 받게 바꾸면 66 x 41 로 되돌려야 한다.
OPTICAL_HFOV_DEG = 51.9     # Camera Module 3, 4:3 으로 자른 뒤
OPTICAL_VFOV_DEG = 40.1
CAM_TO_LEAF_MM = 73.7       # 렌즈 앞면 ~ 리프가 움직이는 평면

CORNER_LABELS = ["1) TOP-LEFT", "2) TOP-RIGHT", "3) BOTTOM-RIGHT", "4) BOTTOM-LEFT"]
CORNER_KO = ["1) 좌상", "2) 우상", "3) 우하", "4) 좌하"]


def grid_shape():
    """조사야 격자 크기 (h, w) [px]."""
    return (int(round(APERTURE_H_MM / MM_PER_PX)),
            int(round(APERTURE_W_MM / MM_PER_PX)))


# ----------------------------------------------------------------------
#  정합 맵
# ----------------------------------------------------------------------

class FieldMap:
    """카메라 픽셀 -> 조사야 격자 로 펴는 변환 하나."""

    def __init__(self, matrix, source, note=""):
        self.M = matrix
        self.source = source          # "calib" 또는 "geometry"
        self.note = note
        self.h, self.w = grid_shape()

    def warp(self, img, interp=cv2.INTER_LINEAR):
        if img is None:
            return None
        return cv2.warpPerspective(img, self.M, (self.w, self.h), flags=interp)

    @property
    def trustworthy(self):
        """융합 판정에 써도 되는가. 기하 추정은 두 카메라를 못 맞춥니다."""
        return self.source == "calib"


def from_points(points):
    """네 귀퉁이 좌표(좌상,우상,우하,좌하) -> FieldMap."""
    h, w = grid_shape()
    src = np.float32(points)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return FieldMap(cv2.getPerspectiveTransform(src, dst), "calib")


def from_geometry(frame_shape, hfov_deg, vfov_deg, dist_mm=CAM_TO_LEAF_MM):
    """보정 파일이 없을 때 화각으로 중앙 크롭을 추정한다.

    카메라가 조사야 정중앙 바로 위에 축을 맞춰 달렸다고 가정합니다.
    실제로는 절대 그렇지 않으므로, 열화상 단독 확인용으로만 쓰세요.
    """
    import math
    fh, fw = frame_shape[0], frame_shape[1]
    field_h = 2.0 * dist_mm * math.tan(math.radians(vfov_deg) / 2.0)
    field_w = 2.0 * dist_mm * math.tan(math.radians(hfov_deg) / 2.0)
    px_per_mm_y = fh / field_h
    px_per_mm_x = fw / field_w

    half_w = APERTURE_W_MM * 0.5 * px_per_mm_x
    half_h = APERTURE_H_MM * 0.5 * px_per_mm_y
    cx, cy = fw * 0.5, fh * 0.5
    pts = [(cx - half_w, cy - half_h), (cx + half_w, cy - half_h),
           (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)]

    h, w = grid_shape()
    src = np.float32(pts)
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    short = (half_w > fw * 0.5 + 0.5) or (half_h > fh * 0.5 + 0.5)
    note = "화면이 조사야보다 좁습니다 (가장자리가 잘립니다)" if short else ""
    return FieldMap(cv2.getPerspectiveTransform(src, dst), "geometry", note)


# ----------------------------------------------------------------------
#  보정 파일
# ----------------------------------------------------------------------

def load_calib(path=CALIB_FILE):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print("[WARN] 보정 파일을 읽지 못했습니다: %s" % exc)
        return None

    ap = data.get("aperture_mm")
    if ap and (abs(ap[0] - APERTURE_W_MM) > 0.1 or abs(ap[1] - APERTURE_H_MM) > 0.1):
        print("[WARN] 보정 파일의 조사야(%.0fx%.0f)가 지금 설정(%.0fx%.0f)과 다릅니다."
              % (ap[0], ap[1], APERTURE_W_MM, APERTURE_H_MM))
        print("       조사야 치수를 바꿨다면 보정을 다시 하세요:  --calib")
    return data


def save_calib(cams, path=CALIB_FILE):
    """cams: {"thermal": [[x,y]x4], "optical": [[x,y]x4]}"""
    data = {
        "aperture_mm": [APERTURE_W_MM, APERTURE_H_MM],
        "mm_per_px": MM_PER_PX,
        "cameras": {k: [[float(p[0]), float(p[1])] for p in v] for k, v in cams.items()},
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "조사야 네 귀퉁이(좌상,우상,우하,좌하)의 각 카메라 픽셀 좌표",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[OK] 보정 저장 — %s" % path)


def map_for(name, data, frame_shape, hfov, vfov):
    """보정 데이터에서 카메라 하나의 FieldMap 을 만든다. 없으면 기하 추정."""
    pts = (data or {}).get("cameras", {}).get(name)
    if pts and len(pts) == 4:
        return from_points(pts)
    return from_geometry(frame_shape, hfov, vfov)


# ----------------------------------------------------------------------
#  4점 보정 UI
# ----------------------------------------------------------------------

def colorize(temp_c, lo=None, hi=None):
    """열화상 float 배열 -> 볼 수 있는 컬러 이미지."""
    a = np.asarray(temp_c, np.float32)
    if lo is None:
        lo = float(np.percentile(a, 1))
    if hi is None:
        hi = float(np.percentile(a, 99))
    g = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)


class _Clicker:
    def __init__(self, scale):
        self.pts = []
        self.scale = float(scale)

    def __call__(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pts) < 4:
            self.pts.append((x / self.scale, y / self.scale))


def _pick_scale(shape, target=640):
    return max(1, int(round(target / float(shape[1]))))


def calibrate_one(read_fn, shape, title, thermal=False, preset=None):
    """카메라 한 대에서 네 귀퉁이를 찍는다. 성공하면 [(x,y)x4], 취소면 None."""
    scale = _pick_scale(shape)
    clicker = _Clicker(scale)
    if preset:
        clicker.pts = [tuple(p) for p in preset]

    win = title
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, clicker)

    print()
    print("=" * 66)
    print(" [%s] 조사야 4점 보정" % title)
    print("=" * 66)
    print(" 1. 콜리메이터를 완전히 개방하고 라이트 필드를 켭니다.")
    print(" 2. 빛의 사각형 네 귀퉁이를 테이프로 표시합니다.")
    if thermal:
        print(" 3. 그 자리에 따뜻한 표식을 하나씩 올려둡니다.")
        print("    (핫팩, 따뜻한 물을 담은 병뚜껑, 손으로 데운 동전 등 무엇이든)")
        print("    ※ 광카메라 보정과 반드시 '같은 표식, 같은 자리'여야 합니다.")
    else:
        print(" 3. 열화상 보정 때 쓴 그 표식을 그대로 둔 채로 찍습니다.")
        print("    두 카메라가 같은 네 점을 보아야 서로 정합됩니다.")
    print(" 4. 화면에서 그 네 점을 순서대로 클릭 :  %s" % "  ->  ".join(CORNER_KO))
    print()
    print("    u = 마지막 점 취소    Enter/s = 확정    q = 취소")
    print("=" * 66)

    result = None
    while True:
        frame = read_fn()
        if frame is None:
            time.sleep(0.02)
            continue
        view = colorize(frame) if thermal else (
            frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        view = cv2.resize(view, (view.shape[1] * scale, view.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST if thermal else cv2.INTER_LINEAR)

        for i, (x, y) in enumerate(clicker.pts):
            p = (int(x * scale), int(y * scale))
            cv2.circle(view, p, 6, (0, 255, 0), 2)
            cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(clicker.pts) == 4:
            cv2.polylines(view, [np.int32([(x * scale, y * scale) for x, y in clicker.pts])],
                          True, (0, 255, 0), 2)
            msg = "4/4  Enter=save  u=undo  q=cancel"
        else:
            msg = "%s   (%d/4)" % (CORNER_LABELS[len(clicker.pts)], len(clicker.pts))
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("u") and clicker.pts:
            clicker.pts.pop()
        if k in (13, 10, ord("s")) and len(clicker.pts) == 4:
            result = list(clicker.pts)
            break

    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return result


def preview_alignment(thermal_grid, optical_grid, n_cols=4):
    """정합 결과를 한 장으로 겹쳐 보여준다. 사람이 눈으로 확인하는 마지막 관문."""
    h, w = grid_shape()
    t = colorize(thermal_grid)
    o = optical_grid if optical_grid.ndim == 3 else cv2.cvtColor(optical_grid, cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(t, 0.5, o, 0.5, 0)
    for c in range(1, n_cols):
        x = (c * w) // n_cols
        cv2.line(blend, (x, 0), (x, h - 1), (255, 255, 255), 1)
    cv2.line(blend, (0, h // 2), (w - 1, h // 2), (0, 255, 255), 1)
    cv2.rectangle(blend, (0, 0), (w - 1, h - 1), (0, 255, 0), 1)
    return blend
