# -*- coding: utf-8 -*-
"""overlay_view.py — 광학 영상을 배경으로 깔고, 열화상이 잡은 체온 경계를 초록 선으로 겹친다.

  [Lepton 3.5]  --USB(Y16)--> 온도 배열 -> 28도 이상 = 인체 마스크 -> 외곽선
                                                                        |
  [Camera Module 3] --CSI--> 광학 영상 <--------- 초록 선으로 겹쳐 그림 ---+

run_row1.py 와 달리 **보기 전용**입니다. 모터를 건드리지 않고 조사야 보정도 쓰지 않습니다.
그래서 카메라를 손에 들고 있어도 쓸 수 있습니다 — 두 카메라가 서로에 대해서만 고정돼
있으면 되기 때문입니다. 반대로, 여기서 맞춘 값으로 리프를 구동해서는 안 됩니다.
리프 구동은 조사야 4점 보정(run_row1.py --calib)이 따로 필요합니다.

정합은 두 카메라의 화각으로 초기값을 잡고, 화면에서 키로 미세조정한 뒤 저장합니다.
저장 파일은 overlay_align.json 입니다.

실행 예)
  python3 overlay_view.py                                  # VNC 화면에서 보며 맞춘다
  python3 overlay_view.py --headless --shot /tmp/ov.png     # SSH 에서 한 장만 저장
  python3 overlay_view.py --tmin 30                        # 임계 온도를 바꿔서 시작

조작키 (VNC 화면에 포커스를 두고)
  q        종료
  방향키 또는 w a s d    초록 선을 상하좌우로 이동
  -  =     초록 선 크기 축소 / 확대 (가로세로 같이)
  [  ]     세로만 축소 / 확대 (가로세로 비율이 안 맞을 때)
  ,  .     체온 임계값 -0.5 / +0.5 도
  b        가장 큰 덩어리만 남기기 on/off (배경의 뜨거운 물체 정리)
  n        상대 기준 on/off — 화면 전체가 사람으로 잡힐 때(센서 예열 중) 쓴다
  c        FFC (열화상 기준 다시 잡기. 온도가 이상하면 이것부터)
  o        열화상만 90도씩 회전 (카메라를 돌려 달았을 때)
  l        overlay_align.json 을 다시 읽어 적용 (프로그램을 끄지 않고 값 교체)
  v        지금 맞춘 정합값을 overlay_align.json 에 저장
  r        정합값을 화각 기본값으로 되돌림
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

import ffc
import leaves
import lepton
import optical as optical_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALIGN_FILE = os.path.join(BASE_DIR, "overlay_align.json")

# 화각 [도]. field.py 와 같은 값을 씁니다.
TH_HFOV, TH_VFOV = 57.0, 44.3      # Lepton 3.5
OP_HFOV, OP_VFOV = 66.0, 41.0      # Camera Module 3 Standard

# body_mask 는 최소 덩어리를 mm^2 로 거릅니다. 여기서는 조사야 격자가 아니라 열화상
# 원본 픽셀 위에서 마스크를 만들므로, 조사야 격자(0.5mm/px)와 같은 비율로 걸리도록
# mm/px 를 환산해 넣습니다. 0.76 이면 약 26px = 열화상 화면의 0.14% 로 같아집니다.
MASK_MM_PER_PX = 0.76


# ----------------------------------------------------------------------
#  정합 (열화상 픽셀 -> 광학 픽셀)
# ----------------------------------------------------------------------

def default_align(th_shape, op_shape):
    """두 카메라가 같은 방향을 본다고 보고 화각만으로 초기 배율을 잡는다.

    렌즈가 6~7cm 떨어져 있어 시차가 남습니다. 그래서 이 값은 어디까지나
    출발점이고, 화면에서 키로 맞춘 뒤 저장해서 쓰는 것이 맞습니다.
    """
    th_h, th_w = th_shape[0], th_shape[1]
    op_h, op_w = op_shape[0], op_shape[1]
    return {
        "sx": (TH_HFOV / OP_HFOV) * (op_w / float(th_w)),
        "sy": (TH_VFOV / OP_VFOV) * (op_h / float(th_h)),
        "tx": 0.0,
        "ty": 0.0,
        "rot": 0.0,
        "tmin_rel": None,
        "tmin_pct": None,
        "tmax": None,
    }


ROT_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_temp(temp_c, rot):
    """열화상을 시계방향으로 rot 도 돌린다 (0/90/180/270).

    카메라를 브래킷에 어떤 방향으로 물리느냐에 따라 열화상만 돌아간 채로 들어옵니다.
    크기·위치만으로는 이걸 못 맞추므로 회전을 정합값의 일부로 둡니다.
    90/270 에서는 가로세로가 바뀌므로 배율(sx/sy)도 같이 다시 잡아야 합니다.
    """
    code = ROT_CODES.get(int(rot) % 360)
    return temp_c if code is None else cv2.rotate(temp_c, code)


def align_matrix(al, th_shape, op_shape):
    """중심을 맞춘 채 배율과 평행이동만 적용하는 2x3 아핀 행렬."""
    th_cy, th_cx = th_shape[0] * 0.5, th_shape[1] * 0.5
    op_cy, op_cx = op_shape[0] * 0.5, op_shape[1] * 0.5
    return np.float32([
        [al["sx"], 0.0, op_cx - th_cx * al["sx"] + al["tx"]],
        [0.0, al["sy"], op_cy - th_cy * al["sy"] + al["ty"]],
    ])


def load_align(th_shape, op_shape):
    al = default_align(th_shape, op_shape)
    al["rot"] = 0.0
    al["tmin_rel"] = None
    al["tmin_pct"] = None
    al["tmax"] = None
    if os.path.exists(ALIGN_FILE):
        try:
            with open(ALIGN_FILE) as fp:
                saved = json.load(fp)
            for k in ("sx", "sy", "tx", "ty", "rot"):
                if k in saved:
                    al[k] = float(saved[k])
            for k in ("tmin_rel", "tmin_pct", "tmax"):   # null 이면 그 방식을 안 쓴다는 뜻
                if k in saved:
                    v = saved[k]
                    al[k] = None if v is None else float(v)
            print("[정합] %s 에서 불러왔습니다 (저장 시각: %s)"
                  % (os.path.basename(ALIGN_FILE), saved.get("saved", "?")))
        except Exception as exc:
            print("[정합] 파일을 못 읽어 기본값을 씁니다: %s" % exc)
    else:
        print("[정합] 저장된 값이 없어 화각으로 추정했습니다. 화면에서 맞춘 뒤 v 로 저장하세요.")
    return al


def align_mtime():
    try:
        return os.path.getmtime(ALIGN_FILE)
    except OSError:
        return None


def save_align(al):
    data = dict(al)
    data["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    data["note"] = "열화상 픽셀 -> 광학 픽셀. overlay_view.py 전용 (조사야 보정과 무관)"
    with open(ALIGN_FILE, "w") as fp:
        json.dump(data, fp, indent=2)
    print("[정합] 저장했습니다 -> %s" % ALIGN_FILE)


# ----------------------------------------------------------------------
#  그리기
# ----------------------------------------------------------------------

def lepton_colorize(temp_c, lo=None, hi=None):
    """열화상을 눈으로 보기 좋게 색칠한다.

    색 범위를 임계 온도(tmin/tmax)에 묶어 두면 안 됩니다. 상한을 80도로 올리는 순간
    화면이 통째로 까매져서 아무것도 판단할 수 없었습니다(2026-08-30). 임계값은
    '어디까지가 사람인가'의 문제이고, 색칠은 '무엇이 보이는가'의 문제라 서로 다릅니다.
    그래서 **그 프레임의 온도 분포**(하위 2% ~ 상위 2%)에 맞춰 자동으로 폅니다.
    센서가 통째로 흘러도 그림은 항상 잘 보입니다.
    """
    if lo is None or hi is None:
        lo = float(np.percentile(temp_c, 2))
        hi = float(np.percentile(temp_c, 98))
    g = np.clip((temp_c - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    return cv2.applyColorMap(g.astype(np.uint8), cv2.COLORMAP_INFERNO)


def effective_tmin(temp_c, tmin, tmin_rel, tmin_pct=None):
    """실제로 쓸 임계 온도. 세 가지 방식 중 하나로 정한다.

      tmin_pct  '화면에서 상위 (100-pct)% 로 뜨거운 곳'을 인체로 본다  ← 가장 튼튼함
      tmin_rel  '화면 중앙값 + N도'
      (없음)    고정 임계값 tmin

    2026-08-30 에 Lepton 이 실온을 24도에서 34도까지 올려 읽는 것을 관측했습니다.
    절대값만 흐르는 게 아니라 **얼굴과 배경의 차이도 9.5도에서 5도로 눌렸습니다.**
    그래서 고정값도, 중앙값+N도도 둘 다 깨집니다 — 둘 다 '온도 차이의 크기'에 기대기
    때문입니다.

    백분위는 그 크기에 기대지 않습니다. 사람이 배경보다 뜨겁다는 순서만 유지되면
    항상 같은 넓이를 잡아냅니다. 우리에게 필요한 것은 절대 온도가 아니라 외곽선입니다.
    다만 화면에 사람이 없어도 무언가는 잡으므로, 인체 유무 판정에는 쓸 수 없습니다.
    """
    if tmin_pct is not None:
        return float(np.percentile(temp_c, float(tmin_pct)))
    if tmin_rel is not None:
        return float(np.median(temp_c)) + tmin_rel
    return tmin


def compose(temp_c, frame_bgr, al, tmin, tmax, thickness=2, pip=True,
            largest=False, relative=False):
    """광학 프레임 위에 체온 외곽선을 초록으로 그려 돌려준다.

    largest 를 켜면 가장 큰 덩어리 하나만 남깁니다. 천장 전등이나 창틀처럼
    사람만큼 따뜻한 배경이 같이 잡힐 때 화면을 정리하는 데 씁니다.
    다만 팔이 몸에서 떨어져 보이면 팔도 같이 사라지므로, 정합을 맞출 때만
    켜고 실제 판정에서는 끄는 것이 안전합니다.

    relative 를 켜면 절대 온도 대신 '이 화면에서 유난히 뜨거운 쪽'을 사람으로 봅니다.
    센서 예열이 덜 끝나 온도가 통째로 흘러도 외곽선은 제대로 나옵니다. 대신 화면에
    사람이 없으면 아무거나 사람으로 잡으므로 **판정용이 아니라 정합·확인용**입니다.
    """
    view = frame_bgr.copy()

    mask = leaves.body_mask(temp_c, tmin=tmin, tmax=tmax,
                            mm_per_px=MASK_MM_PER_PX, largest_only=largest,
                            relative=relative)

    M = align_matrix(al, temp_c.shape, view.shape)
    warped = cv2.warpAffine(mask, M, (view.shape[1], view.shape[0]),
                            flags=cv2.INTER_NEAREST, borderValue=0)

    cnts, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, cnts, -1, (0, 255, 0), thickness)

    if pip:
        # 열화상이 실제로 뭘 보는지 오른쪽 아래에 작게 띄운다. 초록 선이 엉뚱한 데
        # 그려질 때, 카메라가 딴 데를 보는 것인지 정합이 틀린 것인지 가르는 데 쓴다.
        h, w = view.shape[:2]
        if h > 140 and w > 180:
            # 회전을 걸면 열화상이 세로로 길어진다. 160x120 에 억지로 늘리면
            # 찌그러져서 방향을 판단할 수 없으므로, 비율을 지켜 넣고 남는 곳은 검게 둔다.
            box = np.zeros((120, 160, 3), np.uint8)
            th_h, th_w = temp_c.shape[:2]
            f = min(160.0 / th_w, 120.0 / th_h)
            sw, sh = max(1, int(th_w * f)), max(1, int(th_h * f))
            small = cv2.resize(lepton_colorize(temp_c), (sw, sh))
            ox, oy = (160 - sw) // 2, (120 - sh) // 2
            box[oy:oy + sh, ox:ox + sw] = small
            view[h - 124:h - 4, w - 164:w - 4] = box
            cv2.rectangle(view, (w - 165, h - 125), (w - 4, h - 4), (255, 255, 255), 1)

    body_px = int(np.count_nonzero(mask))
    ratio = 100.0 * body_px / mask.size
    lines = [
        "min %.1f  max %.1f C   body %.1f%%%s"
        % (temp_c.min(), temp_c.max(), ratio,
           ("  [rel]" if relative else "") + ("  [largest]" if largest else "")),
        "tmin %.1f C   scale %.2f/%.2f   shift %+.0f/%+.0f"
        % (tmin, al["sx"], al["sy"], al["tx"], al["ty"]),
    ]
    for i, s in enumerate(lines):
        y = 20 + i * 20
        cv2.putText(view, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view, ratio


# ----------------------------------------------------------------------

def wait_for(fn, name, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v is not None:
            return v
        time.sleep(0.05)
    print("!! %s 프레임을 못 받았습니다." % name)
    return None


def cleanup(th, op):
    for obj, meth in ((th, "close"), (op, "release")):
        try:
            getattr(obj, meth)()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermal", default="/dev/mlc_thermal")
    ap.add_argument("--tmin", type=float, default=leaves.TEMP_MIN_C)
    ap.add_argument("--tmax", type=float, default=leaves.TEMP_MAX_C)
    ap.add_argument("--no-ffc", action="store_true", help="시작할 때 FFC 를 걸지 않는다")
    ap.add_argument("--headless", action="store_true", help="창 없이 한 장만 저장하고 끝낸다")
    ap.add_argument("--shot", default="/tmp/overlay.png")
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--largest", action="store_true",
                    help="가장 큰 덩어리만 남긴다 (배경의 뜨거운 물체 제거)")
    ap.add_argument("--relative", action="store_true",
                    help="절대 온도 대신 화면에서 유난히 뜨거운 쪽을 사람으로 본다")
    ap.add_argument("--tmin-rel", type=float, default=None,
                    help="화면 중앙값보다 N도 높은 곳을 사람으로 본다 (센서 드리프트에 강함)")
    ap.add_argument("--win-w", type=int, default=940, help="창 가로 크기 [px]")
    ap.add_argument("--win-h", type=int, default=705, help="창 세로 크기 [px]")
    args = ap.parse_args()
    largest = args.largest
    relative = args.relative

    if not args.no_ffc:
        node = ffc.device_node(args.thermal)
        if node and ffc.ensure_mapped(node):
            print("[FFC] 실행:", ffc.run_once(node))
            time.sleep(1.5)
        else:
            print("[FFC] 매핑되지 않아 건너뜁니다 (온도가 흐르면 c 키로 다시 시도)")

    th = lepton.open_thermal(args.thermal, allow_agc=False, force_agc=False)
    op = optical_mod.OpticalCapture(size=(640, 480))
    time.sleep(1.0)

    temp = wait_for(th.read_celsius, "열화상")
    frame = wait_for(op.read_bgr, "광카메라")
    if temp is None or frame is None:
        cleanup(th, op)
        return 1

    al = load_align(temp.shape, frame.shape)
    if al["tmin_rel"] is None:
        al["tmin_rel"] = args.tmin_rel       # 파일에 없으면 명령줄 값을 쓴다
    tmin, tmax = args.tmin, args.tmax

    if args.headless:
        temp = rotate_temp(temp, al["rot"])
        view, ratio = compose(temp, frame, al,
                              effective_tmin(temp, tmin, al["tmin_rel"], al["tmin_pct"]),
                          al["tmax"] if al["tmax"] is not None else tmax,
                              pip=not args.no_pip, largest=largest, relative=relative)
        cv2.imwrite(args.shot, view)
        print("저장: %s   인체로 잡힌 면적 %.1f%%" % (args.shot, ratio))
        cleanup(th, op)
        return 0

    print(__doc__.split("조작키")[-1])
    win = "MLC overlay  (optical + thermal contour)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # 파이에 달린 화면도, VNC 로 보는 화면도 작다. 기본 640x480 으로 뜨면 창틀에
    # 눌려 실제로는 300px 남짓으로 보여서 초록 선을 맞출 수가 없다.
    cv2.resizeWindow(win, args.win_w, args.win_h)
    cv2.moveWindow(win, 0, 0)

    seen_mtime = align_mtime()
    while True:
        # 파일이 바뀌면 스스로 다시 읽는다. 값을 바꿀 때마다 프로그램을 껐다 켜면
        # 열화상을 여닫게 되고, 그걸 반복하면 Lepton 이 멈춘다 (2026-08-30 두 번 겪음).
        m = align_mtime()
        if m is not None and m != seen_mtime:
            seen_mtime = m
            time.sleep(0.05)          # 쓰는 도중에 읽지 않도록 아주 잠깐 기다린다
            al = load_align(temp.shape, frame.shape)

        t = th.read_celsius()
        f = op.read_bgr()
        if t is not None:
            temp = rotate_temp(t, al["rot"])
        if f is not None:
            frame = f
        view, _ = compose(temp, frame, al,
                          effective_tmin(temp, tmin, al["tmin_rel"], al["tmin_pct"]),
                          al["tmax"] if al["tmax"] is not None else tmax,
                          pip=not args.no_pip, largest=largest, relative=relative)
        cv2.imshow(win, view)

        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            break
        elif k in (ord("a"), 81):
            al["tx"] -= 5
        elif k in (ord("d"), 83):
            al["tx"] += 5
        elif k in (ord("w"), 82):
            al["ty"] -= 5
        elif k in (ord("s"), 84):
            al["ty"] += 5
        elif k in (ord("="), ord("+")):
            al["sx"] *= 1.03
            al["sy"] *= 1.03
        elif k in (ord("-"), ord("_")):
            al["sx"] /= 1.03
            al["sy"] /= 1.03
        elif k == ord("]"):                 # 세로만 늘림 (가로세로 비율 맞추기)
            al["sy"] *= 1.03
        elif k == ord("["):
            al["sy"] /= 1.03
        elif k == ord(","):
            if al["tmin_rel"] is None:
                tmin = max(15.0, tmin - 0.5)
            else:
                al["tmin_rel"] = max(0.5, al["tmin_rel"] - 0.5)
        elif k == ord("."):
            if al["tmin_rel"] is None:
                tmin = min(tmax - 0.5, tmin + 0.5)
            else:
                al["tmin_rel"] = min(15.0, al["tmin_rel"] + 0.5)
        elif k == ord("o"):                 # 열화상만 90도씩 돌린다
            al["rot"] = (al["rot"] + 90) % 360
            temp = rotate_temp(temp, 90)
            print("[회전] 열화상 %d 도" % al["rot"])
        elif k == ord("c"):
            node = ffc.device_node(args.thermal)
            print("[FFC]", ffc.run_once(node) if node else "장치를 못 찾음")
        elif k == ord("b"):
            largest = not largest
            print("[마스크] 가장 큰 덩어리만: %s" % ("on" if largest else "off"))
        elif k == ord("n"):
            relative = not relative
            print("[마스크] 상대 기준(예열 중일 때): %s" % ("on" if relative else "off"))
        elif k == ord("l"):
            # 켜 둔 채로 파일에서 값을 다시 읽는다. 값을 바꾸려고 프로그램을 껐다
            # 켜면 열화상을 여닫게 되는데, 그걸 반복하면 Lepton 이 멈춘다
            # (2026-08-30 에 두 번 겪음). 그래서 재시작 없이 갈아끼울 길을 둔다.
            al = load_align(temp.shape, frame.shape)
        elif k == ord("v"):
            save_align(al)
        elif k == ord("r"):
            al = default_align(temp.shape, frame.shape)
            print("[정합] 기본값으로 되돌렸습니다")

    cv2.destroyAllWindows()
    print("\n마지막 값:  sx=%.4f sy=%.4f tx=%.1f ty=%.1f   tmin=%.1f"
          % (al["sx"], al["sy"], al["tx"], al["ty"], tmin))
    print("저장하려면 실행 중에 v 를 누르세요 (지금 값은 저장되지 않았습니다).")
    cleanup(th, op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
