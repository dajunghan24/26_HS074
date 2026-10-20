# -*- coding: utf-8 -*-
"""
run_row1.py — 라즈베리파이 5 에서 사람 체온 외곽선을 따라 MLC 1행(리프 4장)을 제어

  [Lepton 3.5 / PT3] --USB(Y16)--┐
                                 ├--> [이 프로그램] --USB Serial--> [Arduino UNO]
  [Camera Module 3]  --CSI-------┘        조사야 격자 정합         AMLA_4CH_Positioning
                                          체온 외곽선 + 광학 엣지        L298N x2
                                          -> 리프 4장 개방거리 [mm]      액츄에이터 x4

MLC 는 최종적으로 2행 4열(리프 8장)이지만, 아두이노가 한 대뿐이라 지금은
상단 뱅크 4장만 구동합니다. 하단 뱅크는 두 번째 아두이노를 붙일 때 같은
코드에 --bank bottom 으로 한 대 더 띄우면 됩니다.

실행 예)
  python3 run_row1.py --list                     # 장치 목록
  python3 run_row1.py --calib                    # 조사야 4점 보정 (처음 한 번, 필수)
  python3 run_row1.py --no-serial                # 모터 없이 화면만
  python3 run_row1.py --mock --no-serial         # 카메라도 없이 로직만
  python3 run_row1.py --port auto                # 실제 운전 (모터는 SPACE 로 켬)
  python3 run_row1.py --port auto --motor-on     # 시작부터 모터 출력 on
  python3 run_row1.py --headless --port auto --motor-on   # SSH 로 화면 없이

조작키
  q  종료             SPACE  모터 출력 on/off        h  전 채널 홈 -> 완전 차폐
  s  비상 정지        f      광학 융합 on/off        b  가장 큰 덩어리만 on/off
  [ ] 최저 체온 -/+   , .    최고 체온 -/+           - = 마진 -/+
  t  커서 지점을 손바닥(33도)으로 보고 온도 오프셋 보정
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

import ffc
import field
import leaves
import lepton
import optical as optical_mod
from mlc_driver import MLC, MLCError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KEEPALIVE_S = 1.0        # 아두이노 워치독(LINK_TIMEOUT_MS=3000)보다 반드시 짧게
SEND_INTERVAL_S = 0.15   # 최소 전송 간격
STAT_INTERVAL_S = 0.5    # 펌웨어 위치 조회 주기
DISPLAY_SCALE = 2        # 190x230 격자 -> 380x460 (7인치 800x480 에 맞음)

# 원점 복귀. 채널별 HOME 을 이 간격으로 던져 돌입 전류만 어긋내고 구동은 겹친다.
HOME_STAGGER_S = 0.4     # 출발 간격 [s]
HOME_SWEEP_S = 10.5      # 한 채널이 끝까지 들어가는 시간 (펌웨어 HOMING_SWEEP_MM=120mm)

_GUI_OK = None


ROT_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(img, rot):
    """열화상을 시계방향으로 rot 도 돌린다 (0/90/180/270).

    카메라를 브래킷에 눕혀 달면 열화상만 돌아간 채로 들어옵니다. 리프는 화면의
    '위쪽'을 기준으로 개방거리를 계산하므로, 돌아간 채로 두면 사람이 좌우로
    움직였는데 리프는 위아래로 반응합니다.
    """
    code = ROT_CODES.get(int(rot) % 360)
    return img if (img is None or code is None) else cv2.rotate(img, code)


def rotate_points(pts, rot, w, h):
    """영상을 돌렸으면 조사야 보정 점도 같이 돌려야 정합이 유지된다.

    w, h 는 **회전 전** 프레임 크기입니다.
    """
    r = int(rot) % 360
    out = []
    for x, y in pts:
        if r == 90:
            out.append([h - 1 - y, x])
        elif r == 180:
            out.append([w - 1 - x, h - 1 - y])
        elif r == 270:
            out.append([y, w - 1 - x])
        else:
            out.append([x, y])
    return out


def have_gui():
    """cv2.imshow 를 쓸 수 있는 환경인지 한 번만 검사해 기억한다."""
    global _GUI_OK
    if _GUI_OK is None:
        if sys.platform.startswith("linux") and not (
                os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            _GUI_OK = False
        else:
            try:
                cv2.namedWindow("__probe__")
                cv2.destroyWindow("__probe__")
                _GUI_OK = True
            except Exception:
                _GUI_OK = False
        if not _GUI_OK:
            print("[INFO] 화면이 없어 헤드리스로 동작합니다 (터미널 출력만).")
    return _GUI_OK


# ----------------------------------------------------------------------
#  시리얼 포트 찾기
# ----------------------------------------------------------------------

_PI_OWN_UARTS = ("/dev/ttyAMA", "/dev/ttyS", "/dev/serial0", "/dev/serial1")


def find_serial_ports():
    """아두이노일 법한 포트를 우선순위대로 나열한다."""
    found = []
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            found.append((p.device, (p.description or "") + " " + (p.manufacturer or "")))
    except Exception:
        for pat in ("/dev/ttyACM*", "/dev/ttyUSB*", "COM*"):
            for dev in glob.glob(pat):
                found.append((dev, ""))

    def rank(item):
        dev, desc = item
        d = (dev + " " + desc).lower()
        if dev.startswith(_PI_OWN_UARTS):
            return 9                       # 파이 본체 UART. 아두이노가 아님.
        if "acm" in d or "arduino" in d:
            return 0
        if "usb" in d or "ch340" in d or "wch" in d or "cp210" in d:
            return 1
        return 5

    found.sort(key=rank)
    return found


def autodetect_port():
    ports = find_serial_ports()
    if not ports:
        print("[ERR] 시리얼 포트를 찾지 못했습니다.")
        print("      아두이노 USB 를 확인하고,  ls /dev/ttyACM*  로 보이는지 확인하세요.")
        print("      Permission denied 라면  sudo usermod -aG dialout $USER  후 재부팅.")
        return None
    print("시리얼 포트 후보:")
    for dev, desc in ports:
        print("   %-16s %s" % (dev, desc))
    print("[OK] %s 를 씁니다." % ports[0][0])
    return ports[0][0]


# ----------------------------------------------------------------------
#  아두이노 연결
# ----------------------------------------------------------------------

class Link:
    """MLC 드라이버를 감싸고, 없어도 프로그램이 돌게 해 준다."""

    def __init__(self, port, n_ch=4, travel_mm=leaves.LEAF_TRAVEL_MM, enabled=True):
        self.n = n_ch
        self.travel_mm = travel_mm
        self.mlc = None
        self.last_send = 0.0
        self.pos = [None] * n_ch
        self.homed = [False] * n_ch
        self.err = None
        if not enabled or not port:
            print("[INFO] 모터 없이 동작합니다 (--no-serial).")
            return
        try:
            self.mlc = MLC(port, verbose=False)
            print("[OK] 아두이노 — %s" % self.mlc.ping())
            for line in self.mlc.cfg():
                if line.startswith("#"):
                    print("      " + line)
            # 펌웨어 쪽 안전장치를 이 프로그램의 기하에 맞춰 잠근다.
            self.mlc.limit(travel_mm)      # 행정 상한 = 조사야 절반
            # 통신이 끊기면 **로드를 최대로 내보내** 조사야를 막는다.
            # 2026-08-30 이전 코드는 0.0 을 보냈는데, 이 장치에서 0.0 은
            # 로드 완전 수축 = 조사야 완전 개방이다. 실패안전이 정확히 반대로
            # 걸려 있어서 파이가 죽으면 조사야가 최대로 열렸다.
            self.mlc.failsafe(travel_mm)
            self.mlc.watchdog(True)
            print("[OK] LIM %.1fmm / FS %.1fmm(완전차폐) / 워치독 ON"
                  % (travel_mm, travel_mm))
        except Exception as exc:
            self.err = str(exc)
            self.mlc = None
            print("[ERR] 아두이노에 붙지 못했습니다: %s" % exc)
            print("      화면만 보고 싶으면 --no-serial 을 쓰세요.")

    @property
    def ok(self):
        return self.mlc is not None

    def home(self, shield_after=True):
        if not self.ok:
            return
        try:
            # HOME 은 로드를 리미트 스위치까지 집어넣어 진짜 0 을 만드는 동작이다.
            # 이 장치에서 그 자리는 **조사야 완전 개방**이므로, 홈이 끝나면
            # 곧바로 차폐 위치로 내보내야 한다. 그냥 두면 조사야가 열린 채 대기한다.
            # 펌웨어의 `HOME`(전 채널)은 채널마다 11초씩 **시차**를 둔다
            # (HOME_STAGGER_MS = 11000, 2A 어댑터의 돌입 전류를 겹치지 않게 하려는 것).
            # 그래서 4채널이면 44초가 걸린다.
            #
            # 그런데 4채널 동시 구동은 2026-08-30 에 실측으로 통과했다
            # (SET 15 15 15 15 -> 1.28초에 전부 도달, 어댑터 발열 없음).
            # 문제는 '동시 구동'이 아니라 '돌입 전류가 겹치는 것'이므로,
            # 채널별 HOME 을 0.4초 간격으로 던져 **출발만 어긋내고 구동은 겹친다.**
            # 44초 -> 약 7초. 펌웨어는 그대로 두고 호스트에서만 바꾼 것이다.
            print("원점 복귀 중 (약 %d초, 리프를 끝까지 넣습니다 = 조사야 완전 개방)..."
                  % int(HOME_SWEEP_S + HOME_STAGGER_S * self.n))
            for i in range(self.n):
                self.mlc.home(ch=i + 1, wait=False)
                time.sleep(HOME_STAGGER_S)
            st = self.mlc.wait_idle(timeout=90)
            self.pos, self.homed = st["pos"], st["homed"]
            print("[OK] 홈 완료 pos=%s homed=%s" % (st["pos"], st["homed"]))
            # shield_after=False 면 **수축 상태에서 멈춘다** (= 조사야 완전 개방).
            # 2026-09-07 사용자 요청: HOME 은 "축을 끝까지 넣는 것" 을 뜻하게 한다.
            # 기동 때는 True 로 불러 차폐로 끝낸다 — 아직 아무도 화면을 안 보고
            # 있을 수 있으므로 실패안전 방향으로 두는 것이 옳다.
            if all(st["homed"]) and shield_after:
                print("완전 차폐 위치로 내보냅니다 (%.1fmm)..." % self.travel_mm)
                self.mlc.set([self.travel_mm] * self.n)
                st = self.mlc.wait_idle(timeout=40)
                self.pos = st["pos"]
                print("[OK] 차폐 완료 pos=%s" % st["pos"])
        except MLCError as exc:
            print("[ERR] 홈 실패: %s" % exc)

    def send(self, targets):
        if not self.ok:
            return
        try:
            self.mlc.set(targets)
            self.last_send = time.time()
        except MLCError as exc:
            print("[ERR] 전송 실패: %s" % exc)

    def poll(self):
        if not self.ok:
            return
        try:
            st = self.mlc.stat()
            self.pos, self.homed = st["pos"], st["homed"]
            self.last_send = time.time()      # STAT 도 워치독을 먹여준다
        except MLCError:
            pass

    def stop(self):
        if self.ok:
            try:
                self.mlc.stop()
            except Exception:
                pass

    def close(self, shield=True):
        if not self.ok:
            return
        try:
            if shield:
                # 차폐 = 로드 최대 전진. 0.0 은 완전 개방이라 정반대다.
                print("종료 전 완전 차폐 위치(%.1fmm)로 되돌립니다..." % self.travel_mm)
                self.mlc.set([self.travel_mm] * self.n)
                self.mlc.wait_idle(timeout=40)
            self.mlc.watchdog(False)
        except Exception:
            pass
        self.mlc.close()


# ----------------------------------------------------------------------
#  화면
# ----------------------------------------------------------------------

def patch_temp(tg, x, y, half=2):
    """한 픽셀이 아니라 주변 (2*half+1)^2 의 중앙값을 쓴다.

    Lepton 은 픽셀 단위 잡음이 커서 한 점만 읽으면 누를 때마다 값이 튄다.
    """
    if tg is None:
        return float("nan")
    h, w = tg.shape[:2]
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(np.median(tg[y0:y1, x0:x1]))


def render(state):
    """조사야 격자 위에 열화상·광학·마스크·리프를 겹쳐 한 장으로 만든다."""
    gh, gw = state["grid"]
    scale = DISPLAY_SCALE

    base = field.colorize(state["thermal"], state["tmin"] - 4, state["tmax"] + 2)
    if state["optical"] is not None:
        o = state["optical"]
        if o.ndim == 2:
            o = cv2.cvtColor(o, cv2.COLOR_GRAY2BGR)
        base = cv2.addWeighted(base, 0.65, o, 0.35, 0)

    view = cv2.resize(base, (gw * scale, gh * scale), interpolation=cv2.INTER_LINEAR)
    h, w = view.shape[:2]
    mid = (gh // 2) * scale

    # --- 인체 윤곽 ---
    m = cv2.resize(state["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, cnts, -1, (0, 255, 0), 1)

    # --- 융합으로 다듬은 외곽선 프로파일 ---
    prof = state["profile"]
    # 인체가 없는 열에서는 선을 끊는다. 이어 버리면 조사야를 가로지르는
    # 가짜 대각선이 생겨서 외곽선이 틀린 것처럼 보인다.
    run = []
    for x in range(gw):
        if prof[x] >= 0:
            run.append((x * scale, int(prof[x]) * scale))
            continue
        if len(run) > 1:
            cv2.polylines(view, [np.int32(run)], False, (0, 255, 255), 2)
        run = []
    if len(run) > 1:
        cv2.polylines(view, [np.int32(run)], False, (0, 255, 255), 2)

    # --- 리프가 가리는 영역 ---
    shade = view.copy()
    for c, open_mm in enumerate(state["targets_col"]):
        x0 = ((c * gw) // len(state["targets_col"])) * scale
        x1 = (((c + 1) * gw) // len(state["targets_col"])) * scale
        tip = mid - int(round(open_mm / state["mm_px"])) * scale
        tip = max(0, min(h - 1, tip))
        cv2.rectangle(shade, (x0, 0), (x1 - 1, tip), (40, 40, 40), -1)
        cv2.line(view, (x0, tip), (x1 - 1, tip), (255, 120, 0), 2)
    view = cv2.addWeighted(shade, 0.55, view, 0.45, 0)

    # --- 격자선 ---
    for c in range(1, len(state["targets_col"])):
        x = ((c * gw) // len(state["targets_col"])) * scale
        cv2.line(view, (x, 0), (x, h - 1), (255, 255, 255), 1)
    cv2.line(view, (0, mid), (w - 1, mid), (0, 200, 255), 1)
    # X선 중심선이 지나는 자리. 온도 오프셋 보정의 기준점이기도 하다.
    ccx, ccy = (gw // 2) * scale, (gh // 2) * scale
    cv2.line(view, (ccx - 7, ccy), (ccx + 7, ccy), (255, 255, 255), 1)
    cv2.line(view, (ccx, ccy - 7), (ccx, ccy + 7), (255, 255, 255), 1)
    cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 255, 0), 1)

    # --- 오른쪽 정보판 ---
    panel = np.zeros((h, 420, 3), np.uint8)
    y = 22

    def line(txt, color=(230, 230, 230), dy=20, sz=0.48):
        nonlocal y
        cv2.putText(panel, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sz, color, 1, cv2.LINE_AA)
        y += dy

    motor = "MOTOR ON" if state["motor"] else "motor off"
    line(motor, (0, 255, 0) if state["motor"] else (0, 165, 255), 24, 0.62)
    line("bank=%s  link=%s" % (state["bank"], "OK" if state["link_ok"] else "-"))
    line("align=%s  fusion=%s (snap %d%%)" % (
        state["align"], "on" if state["fusion"] else "off",
        int(state["snap"] * 100)),
        (230, 230, 230) if state["align"] == "calib" else (0, 165, 255))
    line("temp %.1f~%.1f C   margin %.1f mm" % (state["tmin"], state["tmax"], state["margin"]))
    line("fps %.1f   drop %d" % (state["fps"], state["dropped"]))
    cc = patch_temp(state["thermal"], gw // 2, gh // 2)
    line("center %.1f C  (xray axis - press t here)" % cc, (160, 200, 255))
    y += 6

    line("CH  col   open    cmd     act", (160, 200, 255), 22, 0.5)
    for i in range(len(state["send"])):
        col = state["map"][i]
        act = state["pos"][i]
        acts = "  -  " if act is None else "%5.1f" % act
        info = state["infos"][col - 1]
        if info["y_top"] is None:
            flag, color = "SHUT", (0, 165, 255)      # 그 열에 인체 없음 -> 완전 차폐
        elif info["clamped"]:
            flag, color = "OVER", (0, 80, 255)       # 인체가 조사야 밖으로 나감
        else:
            flag, color = "    ", (230, 230, 230)
        line("%2d  %2d  %6.1f  %6.1f  %s %s" % (
            i + 1, col, state["raw_col"][col - 1], state["send"][i], acts, flag), color)
    y += 6
    if any(s["clamped"] for s in state["infos"]):
        line("OVER = body reaches past the aperture edge;", (0, 80, 255), 17, 0.42)
        line("       that part cannot be trimmed at all.", (0, 80, 255), 18, 0.42)
    line("SHUT = no body in that column -> full shield", (150, 150, 150), 18, 0.42)
    line("q quit  SPACE motor  h home  s stop  c ffc", (150, 150, 150), 18, 0.42)
    line("f fusion  b blob  [ ] , . temp  - = margin", (150, 150, 150), 18, 0.42)

    return np.hstack([view, panel])


# ----------------------------------------------------------------------
#  보정 모드
# ----------------------------------------------------------------------

def do_calibration(therm, opt):
    if not have_gui():
        print("[ERR] 4점 보정은 마우스로 찍어야 해서 화면이 필요합니다.")
        print("      파이에 모니터를 붙이거나 VNC 로 접속한 뒤 다시 실행하세요.")
        return 1

    old = field.load_calib() or {}
    cams = {}

    pts = field.calibrate_one(therm.read_celsius, therm.shape, "thermal",
                              thermal=True, preset=old.get("cameras", {}).get("thermal"))
    if pts is None:
        print("취소했습니다.")
        return 1
    cams["thermal"] = pts

    if opt is not None:
        pts = field.calibrate_one(opt.read_bgr, opt.shape, "optical",
                                  thermal=False, preset=old.get("cameras", {}).get("optical"))
        if pts is None:
            print("광카메라 보정을 건너뜁니다. 융합 없이 열화상만 쓰게 됩니다.")
        else:
            cams["optical"] = pts

    field.save_calib(cams)

    # 사람이 눈으로 확인하는 마지막 관문
    tm = field.from_points(cams["thermal"])
    om = field.from_points(cams["optical"]) if "optical" in cams else None
    print()
    print("정합 결과를 겹쳐 보여줍니다. 두 영상의 표식이 겹치면 성공입니다. 아무 키나 누르면 끝.")
    while True:
        t = therm.read_celsius()
        if t is None:
            continue
        tg = tm.warp(t)
        og = om.warp(opt.read_bgr()) if om is not None else np.zeros(tg.shape + (3,), np.uint8)
        cv2.imshow("alignment check", cv2.resize(
            field.preview_alignment(tg, og), None, fx=2, fy=2))
        if cv2.waitKey(30) != -1:
            break
    cv2.destroyAllWindows()
    return 0


# ----------------------------------------------------------------------
#  메인
# ----------------------------------------------------------------------

def parse_map(text):
    try:
        cols = [int(v) for v in text.split(",")]
    except ValueError:
        cols = []
    if sorted(cols) != list(range(1, leaves.N_COLS + 1)):
        raise SystemExit("--map 은 1~%d 을 한 번씩 쓴 목록이어야 합니다. 예: 1,2,3,4"
                         % leaves.N_COLS)
    return cols


def main():
    ap = argparse.ArgumentParser(description="열화상+광학 융합 MLC 1행 제어")
    ap.add_argument("--list", action="store_true", help="카메라 목록만 보고 끝")
    ap.add_argument("--calib", action="store_true", help="조사야 4점 보정")
    ap.add_argument("--thermal", default=None, help="열화상 장치 (예 /dev/video0)")
    ap.add_argument("--optical-index", default=None,
                    help="광카메라를 OpenCV 로 열 때의 인덱스 (picamera2 실패 시)")
    ap.add_argument("--opencv-optical", action="store_true",
                    help="picamera2 대신 OpenCV 로 광카메라를 연다")
    ap.add_argument("--port", default="auto", help="아두이노 포트. auto 면 자동 탐색")
    ap.add_argument("--no-serial", action="store_true", help="모터 없이 화면만")
    ap.add_argument("--motor-on", action="store_true", help="시작부터 모터 출력 on")
    ap.add_argument("--force-uncalibrated", action="store_true",
                    help="보정 없이도 모터를 허용 (위험. 벤치 점검 전용)")
    ap.add_argument("--mock", action="store_true", help="카메라 없이 가짜 영상으로")
    ap.add_argument("--agc", action="store_true", help="열화상을 8비트 AGC 모드로 강제")
    ap.add_argument("--fusion", choices=("on", "off"), default="on")
    ap.add_argument("--bank", choices=("top", "bottom"), default="top",
                    help="이 아두이노가 맡은 뱅크. bottom 이면 영상을 뒤집어 같은 계산을 씁니다")
    ap.add_argument("--map", default="1,2,3,4",
                    help="채널 1~4 가 각각 맡는 열 번호 (왼쪽부터 1). 예 4,3,2,1")
    ap.add_argument("--margin", type=float, default=leaves.MARGIN_MM, help="보호 마진 [mm]")
    ap.add_argument("--ffc-period", type=float, default=60.0,
                    help="열화상 FFC 자동 실행 주기 [초]. 0 이면 끔. "
                         "잔상과 절대온도 흐름을 막습니다")
    ap.add_argument("--tmin", type=float, default=leaves.TEMP_MIN_C,
                    help="인체로 볼 최저 체온 [C]. 화면에서 [ ] 로 맞춘 값을 여기 고정")
    ap.add_argument("--tmax", type=float, default=leaves.TEMP_MAX_C,
                    help="인체로 볼 최고 체온 [C]. 화면에서 , . 로 맞춘 값을 여기 고정")
    ap.add_argument("--deadband", type=float, default=2.5,
                    help="액츄에이터 목표가 이만큼 안 바뀌면 안 보낸다 [mm]. 잡음보다 크게")
    ap.add_argument("--ema", type=float, default=0.25,
                    help="시간 평활. 작을수록 부드럽고 느리다 (닫는 방향에만 걸림)")
    ap.add_argument("--rot", type=int, default=0, choices=(0, 90, 180, 270),
                    help="열화상을 시계방향으로 이만큼 돌려서 쓴다 (카메라를 눕혀 달았을 때)")
    ap.add_argument("--tmin-pct", type=float, default=None,
                    help="상위 (100-N)%% 로 뜨거운 곳을 인체로 본다. 센서 드리프트에 강함")
    ap.add_argument("--headless", action="store_true", help="화면 없이 터미널로만")
    args = ap.parse_args()

    if args.list:
        lepton.list_cameras()
        return 0

    ch_map = parse_map(args.map)
    lepton.load_offset()

    # ---- 카메라 ----
    if args.mock:
        therm = lepton.MockThermal()
        opt = optical_mod.MockOptical() if args.fusion == "on" else None
    else:
        therm = lepton.LeptonCapture(lepton.resolve_cam(args.thermal), force_agc=args.agc)
        opt = None
        if args.fusion == "on":
            try:
                opt = optical_mod.OpticalCapture(
                    index=args.optical_index, prefer_cv2=args.opencv_optical)
            except Exception as exc:
                print(exc)
                print("[WARN] 광카메라 없이 열화상만으로 계속합니다.")

    try:
        if args.calib:
            return do_calibration(therm, opt)

        # ---- 정합 ----
        calib = field.load_calib()

        # 열화상을 돌려서 쓸 때는 보정 점과 화면 크기도 같이 돌려 놓아야 한다.
        th_shape = therm.shape
        rot = int(args.rot) % 360
        if rot:
            th_h, th_w = therm.shape[0], therm.shape[1]
            if rot % 180 == 90:
                th_shape = (th_w, th_h)
            if calib and "cameras" in calib and "thermal" in calib["cameras"]:
                calib["cameras"]["thermal"] = rotate_points(
                    calib["cameras"]["thermal"], rot, th_w, th_h)
            print("[회전] 열화상을 시계방향 %d 도 돌려서 씁니다." % rot)

        tmap = field.map_for("thermal", calib, th_shape,
                             field.THERMAL_HFOV_DEG, field.THERMAL_VFOV_DEG)
        omap = None
        if opt is not None:
            omap = field.map_for("optical", calib, opt.shape,
                                 field.OPTICAL_HFOV_DEG, field.OPTICAL_VFOV_DEG)

        align = "calib" if tmap.trustworthy else "geometry"
        fusion = (opt is not None and omap is not None
                  and tmap.trustworthy and omap.trustworthy)

        print()
        print("=" * 66)
        print(" 조사야 %.0f x %.0f mm,  격자 %d x %d px @ %.2f mm/px"
              % (field.APERTURE_W_MM, field.APERTURE_H_MM,
                 field.grid_shape()[1], field.grid_shape()[0], field.MM_PER_PX))
        print(" 뱅크 %s,  리프 %d장,  액츄에이터 0~%.1f mm"
              " (0 = 조사야 완전 개방, %.1f = 완전 차폐)"
              % (args.bank, leaves.N_COLS,
                 leaves.LEAF_TRAVEL_MM, leaves.LEAF_TRAVEL_MM))
        print(" 정합 %s,  융합 %s" % (align, "ON" if fusion else "OFF"))
        if align != "calib":
            print()
            print(" !! 보정 파일이 없어 화각으로 추정했습니다. 두 카메라가 서로 맞지 않으므로")
            print("    융합을 껐습니다. 실제로 쓰기 전에  --calib  를 반드시 한 번 도세요.")
        if tmap.note:
            print(" !! %s" % tmap.note)
        print("=" * 66)
        print()

        # ---- 아두이노 ----
        port = None if args.no_serial else (
            autodetect_port() if args.port == "auto" else args.port)
        link = Link(port, enabled=not args.no_serial)
        if link.ok:
            link.home()

        gh, gw = field.grid_shape()
        solver = leaves.LeafSolver(gh, gw, field.MM_PER_PX,
                                   fusion=fusion, relative=getattr(therm, "agc", False))
        solver.margin = args.margin
        solver.tmin, solver.tmax = args.tmin, args.tmax
        if args.tmin_pct is not None and args.tmax <= leaves.TEMP_MAX_C:
            # 하한을 백분위로 잡을 때 상한 41도를 그대로 두면 폭이 몇 도밖에 안 되는
            # 좁은 띠가 되어, 화소가 그 띠를 들락날락하며 마스크가 요동친다.
            # 2026-08-30 에 이것 때문에 모터가 쉬지 않고 움직였다. 상한을 열어 둔다.
            solver.tmax = 200.0
            print("[INFO] --tmin-pct 를 쓰므로 상한을 %.0f 도로 열었습니다"
                  " (좁은 띠 요동 방지)." % solver.tmax)
        leaves.EMA_ALPHA = args.ema

        # 잔상 방지. FFC 가 안 돌면 절대 온도까지 흘러서 인체 판정이 무의미해진다.
        ffc_timer = ffc.Periodic(ffc.device_node(args.thermal or "/dev/mlc_thermal"),
                                 args.ffc_period)
        ffc_timer.start()

        motor_allowed = (align == "calib") or args.force_uncalibrated
        if link.ok and not motor_allowed:
            print(" !! 보정이 없어 모터 출력을 잠급니다. --calib 를 먼저 도세요.")
            print("    (벤치 점검이라 그래도 돌려야 한다면 --force-uncalibrated)")
        motor = bool(args.motor_on) and link.ok and motor_allowed
        gui = (not args.headless) and have_gui()
        win = "MLC row1"
        # 마우스를 안 건드리면 X선 중심점이 기준이 된다. 예전 기본값 [0,0]
        # 은 화면 구석이라, 커서를 옮기지 않고 t 를 누르면 벽 온도로 보정됐다.
        cursor = [gw // 2, gh // 2]
        if gui:
            cv2.namedWindow(win)
            cv2.setMouseCallback(
                win, lambda e, x, y, f, p: cursor.__setitem__(slice(0, 2),
                                                              [x // DISPLAY_SCALE,
                                                               y // DISPLAY_SCALE]))

        print("시작합니다.  %s" % ("SPACE 로 모터를 켜세요." if link.ok and not motor
                                   else "q 로 종료합니다."))
        t_last_send = 0.0
        t_last_stat = 0.0
        t_last_print = 0.0
        fps, t_prev = 0.0, time.time()

        while True:
            temp = therm.read_celsius()
            if temp is None:
                if not gui and time.time() - t_last_print > 2:
                    print("[WARN] 열화상 프레임이 오지 않습니다.")
                    t_last_print = time.time()
                continue

            temp = rotate_frame(temp, args.rot)
            tg = tmap.warp(temp)

            if args.tmin_pct is not None:
                # 화면에서 상위 (100-pct)% 로 뜨거운 곳을 인체로 본다.
                # Lepton 이 데워지면 절대 온도가 통째로 흘러서(2026-08-30 실온을
                # 37도로 읽는 것을 관측) 고정 임계값이 무의미해진다. 백분위는
                # 온도 차이의 '크기'에 기대지 않으므로 드리프트를 통째로 우회한다.
                solver.tmin = float(np.percentile(tg, args.tmin_pct))

            og = omap.warp(opt.read_bgr()) if (fusion and opt is not None) else None
            if args.bank == "bottom":
                # 하단 뱅크는 위아래만 뒤집으면 상단 뱅크와 완전히 같은 문제가 된다.
                tg = np.flipud(tg).copy()
                og = None if og is None else np.flipud(og).copy()

            out = solver.solve(tg, og if solver.fusion else None)

            # 열 순서 -> 채널 순서
            send = [out["targets"][c - 1] for c in ch_map]

            now = time.time()
            due = (now - t_last_send >= SEND_INTERVAL_S and solver.changed_enough(out["targets"], args.deadband)) \
                or (now - link.last_send >= KEEPALIVE_S)
            if motor and link.ok and due:
                link.send(send)
                solver.mark_sent(out["targets"])
                t_last_send = now
            if link.ok and now - t_last_stat >= STAT_INTERVAL_S:
                link.poll()
                t_last_stat = now

            dt = now - t_prev
            t_prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            if gui:
                st = {
                    "grid": (gh, gw), "thermal": tg, "optical": og,
                    "mask": out["mask"], "profile": out["profile"],
                    "raw_col": out["raw"], "targets_col": out["targets"],
                    "infos": out["infos"], "send": send, "map": ch_map,
                    "pos": link.pos, "motor": motor, "link_ok": link.ok,
                    "align": align, "fusion": solver.fusion, "snap": solver.snap_ratio,
                    "tmin": solver.tmin, "tmax": solver.tmax, "margin": solver.margin,
                    "fps": fps, "dropped": therm.stats()[1], "mm_px": field.MM_PER_PX,
                    "bank": args.bank,
                }
                cv2.imshow(win, render(st))
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 255
                if now - t_last_print >= 1.0:
                    t_last_print = now
                    print("open %s mm   cmd %s   act %s   %s" % (
                        " ".join("%5.1f" % v for v in out["targets"]),
                        " ".join("%5.1f" % v for v in send),
                        " ".join("  -  " if p is None else "%5.1f" % p for p in link.pos),
                        "MOTOR" if motor else "idle"))

            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                if not motor and not motor_allowed:
                    print("보정이 없어 모터를 켤 수 없습니다. --calib 를 먼저 도세요.")
                    continue
                motor = not motor and link.ok
                print("모터 출력 %s" % ("ON" if motor else "off"))
                if not motor:
                    link.stop()
            elif k == ord("h"):
                link.home()
            elif k == ord("c"):
                print("FFC 실행 (셔터가 1초쯤 닫힙니다)...")
                ffc.run_once(ffc_timer.node)
            elif k == ord("s"):
                motor = False
                link.stop()
                print("비상 정지. SPACE 로 다시 켜세요.")
            elif k == ord("f"):
                if fusion:
                    solver.fusion = not solver.fusion
                    print("융합 %s" % ("ON" if solver.fusion else "off"))
                else:
                    print("이 실행에서는 융합을 쓸 수 없습니다 (보정 또는 광카메라 없음).")
            elif k == ord("b"):
                solver.largest_only = not solver.largest_only
                print("가장 큰 덩어리만 = %s" % solver.largest_only)
            elif k == ord("["):
                solver.tmin -= 0.5
            elif k == ord("]"):
                solver.tmin += 0.5
            elif k == ord(","):
                solver.tmax -= 0.5
            elif k == ord("."):
                solver.tmax += 0.5
            elif k == ord("-"):
                solver.margin = max(0.0, solver.margin - 0.5)
            elif k in (ord("="), ord("+")):
                solver.margin += 0.5
            elif k == ord("t") and gui:
                x, y = cursor
                if 0 <= x < gw and 0 <= y < gh:
                    cur = patch_temp(tg, x, y)
                    lepton.save_offset(lepton.TEMP_OFFSET_C + (lepton.REF_PALM_C - cur))
                    print("온도 오프셋 %+.1f C 로 저장 (커서 %.1f -> %.1f C)"
                          % (lepton.TEMP_OFFSET_C, cur, lepton.REF_PALM_C))

    except KeyboardInterrupt:
        print("\n중단했습니다.")
    finally:
        try:
            ffc_timer.stop()
        except Exception:
            pass
        try:
            print()
            print("이번에 쓴 값:  --tmin %.1f --tmax %.1f --margin %.1f"
                  % (solver.tmin, solver.tmax, solver.margin))
            print("(화면에서 맞춘 값은 저장되지 않습니다. 다음 실행에 위 인자를 붙이세요)")
        except Exception:
            pass
        try:
            link.close(shield=True)
        except Exception:
            pass
        therm.release()
        if opt is not None:
            opt.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
