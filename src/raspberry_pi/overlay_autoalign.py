# -*- coding: utf-8 -*-
"""overlay_autoalign.py — 열화상 외곽선과 광학 영상을 자동으로 맞춘다.

overlay_view.py 에서 방향키로 하던 일을 대신 합니다. 카메라를 브래킷에 다시 물리거나
위치를 바꿨을 때 이걸 한 번 돌리면 됩니다.

원리
  광학 영상에서 밝기가 급변하는 곳(사람 윤곽, 머리카락과 배경의 경계)을 미리 구해 둡니다.
  그다음 열화상 외곽선을 배율과 위치를 바꿔가며 얹어 보고, 그 경계 위에 가장 잘
  포개지는 조합을 고릅니다. 외곽선 점을 직접 옮겨 점수만 매기므로 후보가 수천 개라도
  몇 초면 끝납니다.

한계
  - 사람이 화면 안에 크게 들어와 있어야 합니다. 멀리 있거나 배경과 밝기가 비슷하면
    맞출 근거가 없습니다.
  - 두 카메라가 **서로** 고정돼 있어야 합니다. 각각 손에 들고 있으면 매 순간 답이 달라집니다.
  - 렌즈 간격 때문에 남는 시차는 원리상 못 없앱니다. 피사체 거리가 크게 바뀌면 조금 어긋납니다.

실행 예)
  python3 overlay_autoalign.py                  # 맞춰서 overlay_align.json 에 저장
  python3 overlay_autoalign.py --no-save        # 결과만 보고 저장은 안 함
  python3 overlay_autoalign.py --shot /tmp/a.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import ffc
import leaves
import lepton
import optical as optical_mod
import overlay_view as ov


def grab_thermal(th, n=12, timeout=10.0):
    """여러 장을 중앙값으로 합쳐 잡음을 줄인다."""
    out, t0 = [], time.time()
    while len(out) < n and time.time() - t0 < timeout:
        f = th.read_celsius()
        if f is not None:
            out.append(f)
        time.sleep(0.02)
    if not out:
        return None
    return np.median(np.stack(out), axis=0)


def edge_strength(frame_bgr, blur=5):
    """광학 영상의 경계 세기 지도. 값이 클수록 '여기가 물체의 가장자리'."""
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    g = cv2.GaussianBlur(g, (blur, blur), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # 외곽선이 경계에 딱 얹히지 않아도 점수를 받게 살짝 번지게 한다.
    # 이게 없으면 점수 지형이 뾰족해서 탐색이 엉뚱한 데 걸린다.
    mag = cv2.GaussianBlur(mag, (9, 9), 0)
    return mag / max(float(mag.max()), 1e-6)


def contour_points(mask, max_pts=1200):
    """마스크의 테두리 점들을 (N,2) float32 로. 열화상 픽셀 좌표계."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
    if len(pts) > max_pts:                      # 고르게 솎아낸다
        pts = pts[np.linspace(0, len(pts) - 1, max_pts).astype(int)]
    return pts


# 화면 안에 남아야 하는 외곽선 점의 최소 비율. 피사체가 광학 화면 가장자리에서
# 잘리는 것은 정상이므로 넉넉히 잡는다. 다만 0 으로 두면 외곽선을 화면 밖으로
# 통째로 밀어내고 남은 몇 점만 강한 경계에 얹는 답이 이기므로 하한은 필요하다.
MIN_INSIDE = 0.35


def score(pts, mag, integ, th_shape, op_shape, sx, sy, tx, ty):
    """외곽선을 옮겨 얹고, **주변보다 얼마나 더 뚜렷한 경계 위에 놓였는지**를 잰다.

    외곽선 위 경계 세기의 평균을, 그 외곽선이 덮는 사각 영역의 평균으로 나눕니다.
    그냥 평균만 쓰면 외곽선을 작게 오그라뜨려 천장 전등처럼 경계가 복잡한 구석에
    얹는 답이 무조건 이깁니다 — 실제로 그렇게 틀렸습니다(2026-08-30). 나눗셈을
    넣으면 복잡한 곳에서는 분모도 같이 커져서 그 수법이 통하지 않습니다.

    integ 는 mag 의 적분영상입니다. 사각 영역 평균을 후보마다 다시 더하지 않고
    네 귀퉁이만 빼서 구하려고 미리 만들어 둡니다.
    """
    th_cy, th_cx = th_shape[0] * 0.5, th_shape[1] * 0.5
    op_cy, op_cx = op_shape[0] * 0.5, op_shape[1] * 0.5
    x = (pts[:, 0] - th_cx) * sx + op_cx + tx
    y = (pts[:, 1] - th_cy) * sy + op_cy + ty

    xi = np.round(x).astype(np.int32)
    yi = np.round(y).astype(np.int32)
    ok = (xi >= 0) & (xi < op_shape[1]) & (yi >= 0) & (yi < op_shape[0])
    n = int(ok.sum())
    if n < len(pts) * MIN_INSIDE:
        return -1.0

    xs, ys = xi[ok], yi[ok]
    on_edge = float(mag[ys, xs].mean())

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = (x1 - x0) * (y1 - y0)
    if area < 2000:                 # 너무 작게 오그라든 답은 아예 버린다
        return -1.0
    around = float(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]) / area
    return on_edge / (around + 1e-6)


def search(pts, mag, th_shape, op_shape, base, span=2.20):
    """성긴 격자로 훑고, 이긴 자리 주변을 촘촘히 다시 훑는다.

    배율 범위를 넓게 열어 둡니다. picamera2 가 640x480 을 만들 때 센서를 잘라 쓰기
    때문에, 카탈로그 화각(66도)으로 계산한 배율이 실제와 크게 다릅니다 — 실제로
    ±10% 로 묶었더니 정답이 범위 밖이었습니다(2026-08-30).
    """
    integ = cv2.integral(mag)
    best = None

    def sweep(scales, txs, tys):
        nonlocal best
        for f in scales:
            sx, sy = base["sx"] * f, base["sy"] * f
            for tx in txs:
                for ty in tys:
                    s = score(pts, mag, integ, th_shape, op_shape, sx, sy, tx, ty)
                    if best is None or s > best[0]:
                        best = (s, sx, sy, tx, ty, f)

    lo, hi = 1.0 / span, span
    sweep(np.linspace(lo, hi, 13),
          np.arange(-260, 261, 8.0),
          np.arange(-180, 181, 8.0))
    s, sx, sy, tx, ty, f = best
    sweep(np.linspace(max(lo, f - 0.10), min(hi, f + 0.10), 11),
          np.arange(tx - 10, tx + 10.1, 2.0),
          np.arange(ty - 10, ty + 10.1, 2.0))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermal", default="/dev/mlc_thermal")
    ap.add_argument("--tmin", type=float, default=leaves.TEMP_MIN_C)
    ap.add_argument("--tmax", type=float, default=leaves.TEMP_MAX_C)
    ap.add_argument("--no-ffc", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--shot", default="/tmp/autoalign.png")
    ap.add_argument("--relative", action="store_true",
                    help="절대 온도 대신 화면에서 유난히 뜨거운 쪽을 사람으로 본다")
    ap.add_argument("--span", type=float, default=2.20,
                    help="배율을 화각 추정값의 몇 배까지 찾을지 (2.20 = 0.45~2.2배)")
    ap.add_argument("--capture", metavar="PREFIX",
                    help="찍은 화면을 PREFIX_thermal.npy / PREFIX_optical.png 로 저장")
    ap.add_argument("--use", metavar="PREFIX",
                    help="카메라를 켜지 않고 저장해 둔 화면으로 계산만 다시 한다")
    args = ap.parse_args()

    th = op = None
    if args.use:
        # 카메라를 여닫을수록 Lepton 이 멈출 위험이 커진다. 정합 알고리즘을
        # 손보며 여러 번 돌릴 때는 찍어 둔 화면으로 계산만 반복한다.
        temp = np.load(args.use + "_thermal.npy")
        frame = cv2.imread(args.use + "_optical.png")
        if frame is None:
            print("!! %s_optical.png 를 못 읽었습니다." % args.use)
            return 1
        print("[저장본] %s 로 계산합니다 (카메라 안 켬)" % args.use)
    else:
        if not args.no_ffc:
            node = ffc.device_node(args.thermal)
            if node and ffc.ensure_mapped(node):
                print("[FFC] 실행:", ffc.run_once(node))
                time.sleep(1.5)

        th = lepton.open_thermal(args.thermal, allow_agc=False, force_agc=False)
        op = optical_mod.OpticalCapture(size=(640, 480))
        time.sleep(1.0)

        temp = grab_thermal(th)
        frame = ov.wait_for(op.read_bgr, "광카메라")
        if temp is None or frame is None:
            ov.cleanup(th, op)
            return 1
        if args.capture:
            np.save(args.capture + "_thermal.npy", temp)
            cv2.imwrite(args.capture + "_optical.png", frame)
            print("[저장본] %s_thermal.npy / _optical.png 로 남겼습니다" % args.capture)

    print("열화상 %.1f ~ %.1f C" % (temp.min(), temp.max()))

    # 정합할 때는 몸통 하나만 쓴다. 배경의 뜨거운 물체가 섞이면 답이 흔들린다.
    mask = leaves.body_mask(temp, tmin=args.tmin, tmax=args.tmax,
                            mm_per_px=ov.MASK_MM_PER_PX, largest_only=True,
                            relative=args.relative)
    ratio = 100.0 * np.count_nonzero(mask) / mask.size
    print("인체로 잡힌 면적: %.1f%%" % ratio)
    if ratio < 3.0:
        print("!! 사람이 너무 작게 잡혔습니다. 카메라 앞으로 더 들어오거나")
        print("   ,(쉼표) 로 체온 기준을 낮춘 뒤 다시 하세요. 지금 기준 %.1f 도" % args.tmin)
        ov.cleanup(th, op)
        return 1

    pts = contour_points(mask)
    if pts is None:
        print("!! 외곽선을 못 찾았습니다.")
        ov.cleanup(th, op)
        return 1
    print("외곽선 점 %d 개로 맞춥니다..." % len(pts))

    mag = edge_strength(frame)
    base = ov.default_align(temp.shape, frame.shape)

    t0 = time.time()
    s, sx, sy, tx, ty, f = search(pts, mag, temp.shape, frame.shape, base, args.span)
    print("\n%.1f 초 걸림" % (time.time() - t0))
    print("화각 기본값 : sx=%.3f sy=%.3f  shift +0/+0" % (base["sx"], base["sy"]))
    print("맞춘 값     : sx=%.3f sy=%.3f  shift %+.0f/%+.0f   (배율 x%.2f)"
          % (sx, sy, tx, ty, f))
    print("일치 점수   : %.4f  (0=전혀 안 맞음, 클수록 좋음)" % s)

    al = {"sx": sx, "sy": sy, "tx": tx, "ty": ty}
    view, _ = ov.compose(temp, frame, al, args.tmin, args.tmax, largest=True,
                         relative=args.relative)
    cv2.putText(view, "AUTO score %.3f" % s, (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(view, "AUTO score %.3f" % s, (8, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(args.shot, view)
    print("결과 그림   : %s" % args.shot)

    if args.no_save:
        print("\n--no-save 라 저장하지 않았습니다.")
    else:
        ov.save_align(al)
        print("overlay_view.py 를 다시 켜면 이 값으로 시작합니다.")
        print("눈으로 보고 조금 어긋나면 방향키로 다듬은 뒤 v 로 다시 저장하세요.")

    ov.cleanup(th, op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
