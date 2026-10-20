# -*- coding: utf-8 -*-
"""mlc_core.py — 판정·구동 코어. 화면은 따로 돌고, 소켓으로 이어 붙인다.

  [Lepton 3.5] --USB(Y16)--> 이 프로그램 --USB Serial--> [Arduino UNO] -> 리프 4장
                                  |
                              TCP 소켓 (127.0.0.1:9977)
                                  |
                        [GUI]  Kivy / mp_env / MediaPipe

**열화상과 아두이노는 이 프로그램만 엽니다.** GUI 는 열화상 장치를 직접 열면 안 됩니다.
장치 하나를 두 프로세스가 열 수 없고, 게다가 mp_env 의 cv2 5.0 은 Y16(절대 온도)이
막혀 있어서 GUI 쪽에서는 온도를 읽을 수도 없습니다. 그래서 열화상 '그림'은 여기서
JPEG 로 만들어 소켓으로 보내 줍니다.

프로세스를 나눈 이유 세 가지 (2026-08-30 확인):
  1. GUI 가 얼어도 리프는 닫혀야 한다 — 워치독이 GUI 와 같은 프로세스에 있으면 안 된다
  2. Kivy 가 사는 mp_env 의 cv2 5.0 은 Y16 을 못 연다 (온도 판정 자체가 불가)
  3. MediaPipe 도 mp_env 에만 있다

⚠️ 장치를 한 번 열면 끝까지 붙들고 있어야 합니다.
   - 아두이노: 재접속하면 DTR 리셋으로 원점(HOME) 기억이 날아간다
   - 열화상: 여닫기를 반복하면 센서가 멈춘다 (2026-08-30 두 번 겪음)

**리프 8장 / 아두이노 두 대** (2026-09-06 부터):
  보드 한 대가 4채널이므로 2행 8리프는 보드 두 대로 나뉜다.
  채널 번호는 **1~4 가 상단 뱅크, 5~8 이 하단 뱅크**이고, 소켓으로 오가는
  목록(open_mm / act_mm / pos_mm / homed / flags)도 모두 그 순서다.
  하단 뱅크는 조사야를 아래에서 위로 좁히므로, 화면을 위아래로 뒤집어
  상단과 똑같은 계산을 한 번 더 돌린다.
  어느 보드가 어느 뱅크인지는 소프트웨어가 알 수 없다 — 리프가 반대로
  움직이면 --port 와 --port2 를 맞바꾼다.

실행 예)
  python3 mlc_core.py --rot 90 --tmin-pct 84            # 실전 (리프 8장)
  python3 mlc_core.py --no-bottom                       # 상단 4장만
  python3 mlc_core.py --no-serial                       # 모터 없이 판정만
  python3 mlc_core.py --no-serial --no-thermal          # 장치 없이 소켓만 (규약 확인용)

반드시 **시스템 python3** 로 실행할 것. mp_env 에서 돌리면 Y16 이 열리지 않습니다.
"""

import argparse
import base64
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

import ffc
import field
import leaves
import lepton
from mlc_driver import MLC, MLCError
from run_row1 import Link, rotate_frame, rotate_points

HOST = "127.0.0.1"
PORT = 9977

KEEPALIVE_S = 1.0        # 아두이노 워치독(3초)보다 반드시 짧게
SEND_INTERVAL_S = 0.15   # 아두이노로 보내는 최소 간격
STAT_INTERVAL_S = 0.2    # 펌웨어 위치 조회 주기. 화면이 실제 위치를 따라가야 해서 짧게.
STATE_HZ = 10.0          # GUI 로 상태를 뿌리는 주기
PREVIEW_HZ = 5.0         # 열화상 미리보기 JPEG 주기 (상태보다 느리게)

# 수동 조작에는 잡음이 없으므로 불감대를 거의 없앤다. 자동일 때만 크게 잡는다.
# 0.3mm 는 펌웨어(POS_DEADBAND_MM)가 무시하기 시작하는 값이라 그 아래는 의미가 없다.
MANUAL_DEADBAND_MM = 0.3

# 기준 프레임을 잡을 때 평균할 장수
REF_FRAMES = 12

# [준비] 를 눌렀을 때 "화면에 사람이 있다"고 볼 기준.
# 가장 뜨거운 쪽 10% 중 **한 덩어리로 이어진 것**이 화면의 이만큼을 넘으면 사람으로 본다.
# 빈 화면에서 뜨거운 쪽 10% 는 잡음이라 흩어져 있고, 사람은 한 덩어리로 뭉친다.
OCCUPIED_PCT = 8.0


def looks_occupied(grid):
    """화면에 사람만 한 것이 있는가. [준비] 를 잘못 누르는 것을 막기 위한 것."""
    mask = leaves.body_mask(grid, mm_per_px=field.MM_PER_PX,
                            relative=True, largest_only=True)
    return 100.0 * float(np.count_nonzero(mask)) / mask.size > OCCUPIED_PCT


# ======================================================================
#  소켓 허브
# ======================================================================

class Hub:
    """GUI 들과 줄 단위 JSON 을 주고받는다.

    한 줄에 JSON 하나입니다. 붙는 쪽이 여러 개여도 됩니다(화면 두 개, 진단용 nc 등).
    끊긴 클라이언트는 조용히 버립니다 — GUI 가 죽어도 코어는 계속 돌아야 합니다.
    """

    def __init__(self, host=HOST, port=PORT):
        self.addr = (host, port)
        self.clients = []
        self.lock = threading.Lock()
        self.inbox = []                  # [(명령 dict, 소켓)]
        self.inbox_lock = threading.Lock()
        self.srv = None

    def start(self):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(self.addr)
        self.srv.listen(5)
        threading.Thread(target=self._accept_loop, daemon=True).start()
        print("[소켓] %s:%d 에서 GUI 를 기다립니다." % self.addr)

    def _accept_loop(self):
        while True:
            try:
                sock, peer = self.srv.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.clients.append(sock)
            print("[소켓] 접속: %s (지금 %d 개)" % (peer[0], len(self.clients)))
            threading.Thread(target=self._read_loop, args=(sock,), daemon=True).start()

    def _read_loop(self, sock):
        buf = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        self.send_to(sock, {"ok": False, "err": "JSON 아님: %s" % exc})
                        continue
                    with self.inbox_lock:
                        self.inbox.append((msg, sock))
        except OSError:
            pass
        finally:
            self._drop(sock)

    def _drop(self, sock):
        with self.lock:
            if sock in self.clients:
                self.clients.remove(sock)
                print("[소켓] 끊김 (남은 %d 개)" % len(self.clients))
        try:
            sock.close()
        except OSError:
            pass

    def send_to(self, sock, obj):
        try:
            sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            self._drop(sock)

    def broadcast(self, obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            targets = list(self.clients)
        for sock in targets:
            try:
                sock.sendall(data)
            except OSError:
                self._drop(sock)

    def pop_commands(self):
        with self.inbox_lock:
            out, self.inbox = self.inbox, []
        return out


# ======================================================================
#  아두이노 두 대 = 리프 8장
# ======================================================================

class BankPair:
    """아두이노 두 대를 링크 한 개처럼 보이게 감싼다.

    채널 번호는 **1~4 가 상단 뱅크, 5~8 이 하단 뱅크**다. 목록으로 오가는 값
    (open_mm / act_mm / pos / homed / flags)은 전부 '상단 4개 다음 하단 4개'
    순서로 이어 붙인다. 코어의 나머지 코드는 `self.link` 하나만 보므로,
    보드가 두 대가 됐다는 사실이 이 클래스 밖으로 새지 않는다.

    보드 한 대가 죽어도 **자리는 그대로 둔다.** 죽은 보드를 목록에서 빼 버리면
    살아남은 보드의 채널 번호가 앞으로 당겨져, 하단 뱅크에 보낼 값이 상단
    뱅크로 가는 사고가 난다. 그래서 `all` 은 전부 들고 있고 전송할 때만 건너뛴다.
    """

    def __init__(self, links):
        self.all = [l for l in links if l is not None]
        self.n = sum(l.n for l in self.all)
        self.travel_mm = self.all[0].travel_mm if self.all else 0.0

    @property
    def live(self):
        return [l for l in self.all if l.ok]

    @property
    def ok(self):
        return bool(self.live)

    @property
    def homed(self):
        return [h for l in self.all for h in l.homed]

    @property
    def pos(self):
        return [p for l in self.all for p in l.pos]

    @property
    def err(self):
        for l in self.all:
            if l.err:
                return l.err
        return None

    @property
    def last_send(self):
        # 가장 오래 굶은 보드를 기준으로 본다. 워치독은 보드마다 따로 돌기
        # 때문에, 평균이나 최댓값을 쓰면 한쪽만 굶다가 그 보드에서 워치독이 터진다.
        return min((l.last_send for l in self.live), default=0.0)

    def home(self, shield_after=True):
        # 보드끼리는 **차례로** 돌린다. 한 보드 안에서는 0.4초 시차로 겹쳐도
        # 되지만(2026-08-30 실측), 8개를 한꺼번에 출발시키면 2A 어댑터의
        # 돌입 전류를 감당하지 못한다.
        #
        # shield_after=False 면 수축에서 멈춘다 (= 조사야 완전 개방).
        for i, l in enumerate(self.live):
            print("[뱅크 %d] 원점 복귀" % (i + 1))
            l.home(shield_after=shield_after)

    def send(self, targets):
        off = 0
        for l in self.all:
            if l.ok:
                l.send(list(targets[off:off + l.n]))
            off += l.n

    def poll(self):
        for l in self.live:
            l.poll()

    def stop(self):
        for l in self.live:
            l.stop()

    def close(self, shield=True):
        for l in self.live:
            try:
                l.close(shield=shield)
            except Exception:
                pass


# ======================================================================
#  코어
# ======================================================================

class Core:
    def __init__(self, args):
        self.args = args
        self.hub = Hub(port=args.port_tcp)
        self.motor = args.motor_on
        self.manual = None          # 수동 개방거리 [mm] x4. None 이면 카메라가 정한다
        self.err = None

        # 기준 프레임 (환자가 없을 때의 화면). 이걸 빼면 잔상과 드리프트가 함께
        # 상쇄된다 — 둘 다 기준 화면에도 똑같이 들어 있기 때문이다.
        # 옷 표면과 배경의 차이는 3~5도밖에 안 되므로, 드리프트를 못 없애면
        # Shoulder Lateral 같은 '옷 입은 부위'는 판정 자체가 불가능하다.
        self.ref = None
        self.recent = []            # 최근 프레임 몇 장 (기준을 평균으로 잡으려고)
        self.preparing = False      # [준비] 진행 중인가
        self.last_center = None     # 중심선 기준일 때 마지막 중심 온도 [℃]
        self.center_ok = None       # 중심에 인체가 있다고 봤는가 (None = 그 모드 아님)
        self.last_temp = None       # 잘라내기 전 전체 화면 (사람 유무 판정용)

        # 광카메라에서 받은 인체 마스크. 이게 있으면 리프 계산의 주(主) 근거가 된다.
        self.outline = None
        self.outline_t = 0.0
        self.omap = None            # 광카메라 -> 조사야 격자 (첫 마스크에서 만든다)
        self._omap_shape = None
        # 광카메라 사진 그 자체. seg_feed 가 마스크와 같이 보내온다.
        # 방사선사가 화면에서 보는 것은 열화상이 아니라 이것이어야 한다 —
        # 사람 눈으로 '환자가 맞게 서 있나'를 판단할 수 있는 그림이기 때문이다.
        self.optical_bgr = None
        self.optical_bgr_t = 0.0
        self.calib = None
        # 채널 수는 뱅크를 열어 보고 확정한다 (open_devices). 여기서는 보드
        # 한 대분을 임시로 넣어 둔다 — 열기 전에 상태를 물어보는 경우가 있다.
        self.n = leaves.N_COLS
        self.travel = leaves.LEAF_TRAVEL_MM
        self.solvers = []
        self.outs = None            # 뱅크별 계산 결과 (열 상태 flags 를 만든다)

    # ---------- 장치 ----------

    def open_devices(self):
        a = self.args

        self.therm = None
        self.ffc = None
        if not a.no_thermal:
            self.therm = lepton.open_thermal(a.thermal, allow_agc=False, force_agc=False)
            # FFC 는 배경 스레드에서 돌린다. 셔터를 여닫느라 2~3초 걸리는데,
            # 본체 루프에서 부르면 그동안 아두이노 워치독(3초)이 발동한다.
            if not a.no_ffc:
                self.ffc = ffc.Periodic(ffc.device_node(a.thermal), a.ffc_period)
                self.ffc.start()

        # ---- 조사야 정합 ----
        calib = field.load_calib()
        th_shape = (120, 160) if self.therm is None else self.therm.shape
        self.rot = int(a.rot) % 360
        if self.rot:
            th_h, th_w = th_shape[0], th_shape[1]
            if self.rot % 180 == 90:
                th_shape = (th_w, th_h)
            # 보정 좌표도 같이 돌린다 = 소프트웨어 회전이 격자 결과에 영향을
            # 주지 않는다(서로 상쇄). 카메라가 물리적으로 안 움직였을 때 맞다.
            #
            # --rot-image-only 는 이 상쇄를 끊는다. **보정 이후 카메라가 광축
            # 중심으로 돌아간 경우**의 임시 대응이다 — 영상만 돌아가므로 격자
            # 내용이 실제로 회전한다. 위치까지 움직였으면 못 잡으니, 제대로 된
            # 해결은 4점 재보정(run_row1.py --calib)이다.
            if (calib and "cameras" in calib and "thermal" in calib["cameras"]
                    and not a.rot_image_only):
                calib["cameras"]["thermal"] = rotate_points(
                    calib["cameras"]["thermal"], self.rot, th_w, th_h)
            print("[회전] 열화상을 시계방향 %d 도 돌려서 씁니다." % self.rot)

        self.calib = calib
        self.tmap = field.map_for("thermal", calib, th_shape,
                                  field.THERMAL_HFOV_DEG, field.THERMAL_VFOV_DEG)
        self.aligned = self.tmap.trustworthy

        gh, gw = field.grid_shape()
        # 뱅크마다 솔버를 따로 둔다.
        #
        # 하단 뱅크는 화면을 위아래로 뒤집으면 상단 뱅크와 **완전히 같은 문제**가
        # 된다 (run_row1.py 의 --bank bottom 이 쓰던 처리 그대로다). 그래서 같은
        # 클래스를 한 번 더 돌린다. 인스턴스를 나누는 이유는 시간 평활(EMA)과
        # '마지막으로 보낸 값'이 뱅크별로 따로 쌓여야 하기 때문이다. 하나를
        # 돌려쓰면 두 뱅크가 서로의 평활 상태를 덮어써서 리프가 튄다.
        self.solvers = [leaves.LeafSolver(gh, gw, field.MM_PER_PX, fusion=False)
                        for _ in range(2 if a.bottom_bank else 1)]
        for s in self.solvers:
            s.margin = a.margin
            s.tmin, s.tmax = a.tmin, a.tmax
        self.solver = self.solvers[0]       # 상단 뱅크. 기존 이름을 그대로 둔다

        # 잡음 때문에 리프가 쉬지 않고 떠는 것을 막는다. 2026-08-30 에 기본값
        # 0.8mm 로는 센서 잡음을 못 이겨 모터가 계속 움직였다.
        leaves.EMA_ALPHA = a.ema

        # ---- 아두이노 ----
        # 보드 한 대가 리프 4장이다. 2행 8리프는 보드 두 대로 나뉘고, 채널 번호는
        # 1~4 상단 / 5~8 하단으로 이어 붙인다.
        #
        # 어느 보드가 어느 뱅크인지는 **소프트웨어가 알 수 없다.** udev 이름
        # (mlc_uno1/mlc_uno2)은 USB 컨트롤러 위치로 붙는 것이라 배선과 무관하다.
        # 기본값은 2026-09-06 기준 실제로 상단 뱅크를 돌리고 있던 보드를
        # --port 로 잡은 것이다. 리프가 반대로 움직이면 두 값을 맞바꾸면 된다.
        n_col = leaves.N_COLS
        enabled = not a.no_serial
        links = [Link(None if a.no_serial else a.port, n_ch=n_col,
                      travel_mm=self.travel, enabled=enabled)]
        if a.bottom_bank:
            links.append(Link(None if a.no_serial else a.port2, n_ch=n_col,
                              travel_mm=self.travel, enabled=enabled))
        self.link = BankPair(links)
        self.n = self.link.n
        print("[뱅크] 리프 %d장 — %s" % (
            self.n, "상단 %s + 하단 %s" % (a.port, a.port2) if a.bottom_bank
            else "상단 %s 만" % a.port))
        if self.link.ok:
            self.link.home()          # 홈 뒤 곧바로 완전 차폐까지 간다

    def optical_map(self, shape):
        """광카메라 화면 -> 조사야 격자 변환. 첫 마스크를 받을 때 한 번 만든다.

        코어는 광카메라를 직접 열지 않는다(GUI 가 연다). 그래서 화면 크기를 미리
        알 수 없고, 마스크가 처음 도착했을 때 그 크기로 만든다.

        보정 파일에 광카메라 네 귀퉁이가 없으면 화각 추정으로 떨어지는데, 그건
        두 카메라를 못 맞춘다. 그때는 `aligned` 가 False 로 보고된다.
        """
        if self.omap is not None and self._omap_shape == shape[:2]:
            return self.omap
        try:
            self.omap = field.map_for("optical", self.calib, shape,
                                      field.OPTICAL_HFOV_DEG, field.OPTICAL_VFOV_DEG)
            self._omap_shape = shape[:2]
            print("[외곽선] 광카메라 %dx%d -> 조사야 격자 (%s)"
                  % (shape[1], shape[0], self.omap.source))
            return self.omap
        except Exception as exc:
            self.err = "광카메라 정합 실패: %s" % exc
            print("[외곽선] 정합 실패:", exc)
            return None

    # ---------- 준비 ----------

    def _prepare(self):
        """FFC 를 걸고 그 깨끗한 상태의 배경을 기준으로 잡는다. 배경 스레드에서 돈다.

        본체 루프에서 하면 셔터를 여닫는 2~3초 동안 아두이노 워치독(3초)이 터진다.
        """
        try:
            node = ffc.device_node(self.args.thermal)
            if node and ffc.ensure_mapped(node):
                print("[준비] FFC 실행:", ffc.run_once(node))
            else:
                print("[준비] FFC 제어를 못 찾아 건너뜁니다 (기준만 잡습니다)")
            time.sleep(1.5)                  # FFC 뒤 프레임이 새로 들어올 시간
            frames = list(self.recent)
            if not frames:
                self.err = "준비 실패: 프레임 없음"
                return
            self.ref = np.median(np.stack(frames), axis=0)
            self.err = None
            print("[준비] 완료 — %d장을 평균해 기준을 잡았습니다." % len(frames))
        except Exception as exc:
            self.err = "준비 실패: %s" % exc
            print("[준비] 실패:", exc)
        finally:
            self.preparing = False

    # ---------- 명령 ----------

    def handle(self, msg, sock):
        cmd = str(msg.get("cmd", "")).lower()
        try:
            if cmd == "ping":
                pass
            elif cmd == "motor":
                self.motor = bool(msg.get("on", False))
                if not self.motor:
                    self.link.stop()
            elif cmd == "home":
                # ⚠️ 운영자가 누르는 HOME 은 **수축에서 멈춘다** (조사야 완전 개방).
                # 2026-09-07 사용자 요청. 기동 때(open_devices)는 차폐로 끝난다.
                # ⚠️ 목표값을 **먼저** 개방으로 바꾼다. home() 은 뱅크를 차례로
                # 도는데, 그동안 본체 루프가 옛 목표(차폐)를 계속 보내면 먼저
                # 끝난 뱅크가 곧바로 다시 밀려난다. 2026-09-07 에 실제로
                # 앞행만 60 으로 되돌아갔다.
                self.manual = [self.travel] * self.n
                self.link.home(shield_after=False)
            elif cmd == "stop":
                self.motor = False
                self.link.stop()
            elif cmd == "shield":                 # 전체 차폐
                self.manual = [0.0] * self.n      # 개방거리 0 = 완전 차폐
            elif cmd == "auto":                   # 카메라 판정으로 복귀
                self.manual = None
            elif cmd == "manual":
                # 개방거리를 사람이 직접 지정한다 (GUI 의 UP/DOWN 버튼용).
                #   ch 를 주면      그 채널만
                #   open_mm 배열이면 4개를 한꺼번에
                #   **둘 다 없으면  지금 값 그대로 멈춘 채 수동으로 전환**
                # 마지막 경우가 AUTO/MANUAL 토글 버튼이 쓰는 것이다. 리프를
                # 움직이지 않고 카메라 제어만 떼어내므로, 지금 자리에서 이어서
                # 미세조정할 수 있다.
                if self.manual is None:
                    self.manual = [v for s in self.solvers for v in s.smoothed]
                if "ch" in msg:
                    i = int(msg["ch"]) - 1
                    if not 0 <= i < self.n:
                        raise ValueError("ch 는 1~%d" % self.n)
                    self.manual[i] = float(np.clip(msg.get("open_mm", 0.0),
                                                   0.0, self.travel))
                elif "open_mm" in msg:
                    v = msg.get("open_mm")
                    if not isinstance(v, (list, tuple)) or len(v) != self.n:
                        raise ValueError("open_mm 은 길이 %d 인 배열" % self.n)
                    self.manual = [float(np.clip(x, 0.0, self.travel)) for x in v]
            elif cmd == "outline":
                # GUI 가 광카메라에서 딴 인체 마스크를 보내온다.
                #
                # 우리에게 필요한 것은 열화상 영상이 아니라 **환자 체형의 외곽선**이다.
                # 광카메라는 640x480 으로 열화상(160x120)보다 4배 정밀하고, 잔상도
                # 드리프트도 없다. MediaPipe 인체 분할이 파이 5 에서 15ms 에 돈다
                # (2026-08-31 실측). 그래서 외곽선은 광카메라가 맡고, 열화상은
                # '살아있는 사람인가'를 확인하는 보조로 돌린다.
                png = msg.get("png")
                if not png:
                    raise ValueError("png 가 없습니다")
                buf = np.frombuffer(base64.b64decode(png), np.uint8)
                m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    raise ValueError("마스크를 못 읽었습니다")
                self.outline = (m > 127).astype(np.uint8) * 255
                self.outline_t = time.time()
                # 사진은 있으면 받고 없으면 그만이다. 마스크만 와도 리프는 돈다.
                cam = msg.get("cam")
                if cam:
                    cbuf = np.frombuffer(base64.b64decode(cam), np.uint8)
                    cimg = cv2.imdecode(cbuf, cv2.IMREAD_COLOR)
                    if cimg is not None:
                        self.optical_bgr = cimg
                        self.optical_bgr_t = time.time()
            elif cmd == "prepare":
                # 방사선사가 **환자를 들이기 전에** 한 번 누르는 것.
                # FFC 로 센서를 가장 깨끗한 상태로 만든 뒤, 그 상태의 배경을
                # 기준 프레임으로 잡는다. 두 작업의 적기가 같아서 하나로 묶었다.
                #
                # 환자가 화면에 있을 때 누르면 **환자를 배경으로 학습**해버려서
                # 그 뒤로 환자를 못 알아본다. 리프가 안 닫히게 되므로 막는다.
                if self.preparing:
                    raise ValueError("이미 준비 중입니다")
                if not self.recent:
                    raise ValueError("아직 열화상 프레임이 없습니다")
                # 판정은 **잘라내기 전 전체 화면**으로 한다. 조사야 격자는 4점 보정으로
                # 좁게 잘라낸 것이라, 거기서는 뭐가 있든 화면의 큰 비율을 차지한다.
                # (2026-08-31: 격자로 판정했더니 사람이 없어도 계속 거부했다)
                if self.last_temp is None:
                    raise ValueError("아직 열화상 프레임이 없습니다")
                if looks_occupied(self.last_temp):
                    raise ValueError("화면에 사람이 있습니다. 비켜주신 뒤 다시 누르세요")
                self.preparing = True
                threading.Thread(target=self._prepare, daemon=True).start()
            elif cmd == "ref":
                # 지금 화면을 '사람 없을 때의 기준'으로 저장한다.
                # 환자가 들어오기 전에 한 번 누르면 된다. 카메라가 움직이거나
                # 배경이 바뀌면 다시 잡아야 한다.
                if not self.recent:
                    raise ValueError("아직 열화상 프레임이 없습니다")
                self.ref = np.median(np.stack(self.recent), axis=0)
                print("[기준] %d장을 평균해 기준 프레임을 잡았습니다." % len(self.recent))
            elif cmd == "ref_clear":
                self.ref = None
                print("[기준] 해제했습니다.")
            elif cmd == "hold_stop":
                # 버튼을 꾹 누르고 있다가 손을 뗐다. **지금 실제 위치에서 멈춘다.**
                #
                # 부드럽게 움직이려면 누르고 있는 동안 목표를 끝까지 줘서 모터가
                # 쉬지 않고 달리게 해야 한다. 1mm 씩 잘게 보내면 매번
                # '출발-구동-정지'를 반복해 뚝뚝 끊긴다.
                # 손을 뗄 때는 펌웨어에서 위치를 **새로 읽어** 그 자리를 목표로
                # 삼는다. 마지막 STAT 캐시를 쓰면 최대 0.5초치를 지나쳐 버린다.
                self.link.poll()
                pos = list(self.link.pos)
                if any(p is None for p in pos):
                    raise ValueError("아직 위치를 모릅니다 (HOME 이 필요합니다)")
                self.manual = [float(np.clip(self.travel - p, 0.0, self.travel))
                               for p in pos]
            elif cmd == "margin":
                for s in self.solvers:          # 두 뱅크에 똑같이 건다
                    s.margin = float(msg["v"])
            elif cmd == "tmin_pct":
                self.args.tmin_pct = None if msg.get("v") is None else float(msg["v"])
            elif cmd == "tmin_center":
                # 중심선 기준 켜기/끄기. v = 중심 온도에서 내릴 폭 [℃], None = 끔.
                # 켜져 있으면 tmin_pct 보다 우선한다. 재시작 없이 되돌릴 수 있게
                # 소켓으로도 열어 둔다 (2026-09-10 시험용).
                self.args.tmin_center = (None if msg.get("v") is None
                                         else float(msg["v"]))
                if self.args.tmin_center is None:
                    self.last_center = None
                    self.center_ok = None
            elif cmd == "tmin":
                self.args.tmin_pct = None
                for s in self.solvers:
                    s.tmin = float(msg["v"])
            elif cmd == "ffc":
                # 셔터를 여닫느라 2~3초 걸린다. 본체 루프를 막으면 워치독이 터지므로
                # 별도 스레드로 던진다.
                node = ffc.device_node(self.args.thermal)
                threading.Thread(target=ffc.run_once, args=(node,),
                                 daemon=True).start()
            else:
                raise ValueError("모르는 명령: %r" % cmd)
            self.hub.send_to(sock, {"ok": True, "cmd": cmd})
        except Exception as exc:
            self.hub.send_to(sock, {"ok": False, "cmd": cmd, "err": str(exc)})

    # ---------- 본체 ----------

    def run(self):
        self.open_devices()
        self.hub.start()
        print("코어 시작. Ctrl-C 로 종료합니다.")

        t_send = t_stat = t_state = t_prev = t_ffc = 0.0
        tg = None
        temp = None
        preview_b64 = None
        t_preview = 0.0

        try:
            while True:
                now = time.time()

                for msg, sock in self.hub.pop_commands():
                    self.handle(msg, sock)

                # ---- 열화상 ----
                if self.therm is not None:
                    t = self.therm.read_celsius()
                    if t is not None:
                        temp = rotate_frame(t, self.rot)
                        self.last_temp = temp
                        tg = self.tmap.warp(temp)

                # 기준 프레임을 잡을 수 있게 최근 몇 장은 항상 들고 있는다.
                # (광카메라 경로로 가더라도 [준비] 는 열화상 기준을 잡아야 한다)
                if tg is not None:
                    self.recent.append(tg)
                    if len(self.recent) > REF_FRAMES:
                        self.recent.pop(0)

                self.source = "thermal"
                out = None

                # ---- 광카메라 외곽선이 있으면 그것을 주(主)로 쓴다 ----
                #
                # 필요한 것은 온도가 아니라 외곽선이다. 광카메라 분할은 640x480 에
                # 잔상·드리프트가 없고 15ms 에 끝난다. 열화상은 160x120 에 잔상이
                # 남는다 (2026-08-31 확인). 그래서 외곽선은 광카메라가 맡는다.
                #
                # 너무 오래된 외곽선은 버린다. GUI 가 죽었는데 마지막 마스크로
                # 계속 리프를 벌려 두면 위험하다 — 사람이 없어졌을 수도 있다.
                fresh = (self.outline is not None
                         and now - self.outline_t < self.args.outline_timeout)
                if fresh:
                    om = self.optical_map(self.outline.shape)
                    gmask = om.warp(self.outline) if om is not None else None
                    if gmask is not None:
                        gmask = (gmask > 127).astype(np.uint8) * 255
                        self.outs, open_mm, act = self.solve_banks(gmask, True)
                        out = self.outs[0]
                        self.source = "optical"
                    else:
                        fresh = False

                if not fresh and tg is not None:
                    if self.ref is not None and self.ref.shape == tg.shape:
                        # **사람 없을 때의 화면을 뺀다.** 잔상(고정 무늬)과
                        # 드리프트(전체가 같이 뜨는 것)가 함께 상쇄된다.
                        work = tg - self.ref
                        # 뺀 뒤에도 남는 균일한 흐름은 중앙값으로 흡수한다.
                        # 임계값은 뱅크와 무관하므로 두 솔버에 똑같이 넣는다.
                        self.set_tmin(float(np.median(work)) + self.args.ref_delta)
                    else:
                        work = tg
                        thr = self.body_threshold(tg)
                        if thr is not None:
                            self.set_tmin(thr)
                    self.outs, open_mm, act = self.solve_banks(work, False)
                    out = self.outs[0]

                    # 기준 프레임을 **사람이 없다고 판단된 곳만** 천천히 따라가게 한다.
                    #
                    # 기준을 한 번 찍고 고정해 두면, 시간이 지나며 센서 상태가 조금씩
                    # 달라져 그 어긋남이 화면에 남는다. 특히 사람이 지나간 자리에
                    # 검은 잔상으로 보인다 (2026-08-31 관측). 배경만 조금씩 다시
                    # 배우면 그 자리가 몇 초 만에 배경으로 돌아와 잔상이 사라진다.
                    #
                    # 사람으로 잡힌 곳은 건드리지 않는다. 안 그러면 가만히 서 있는
                    # 환자를 배경으로 학습해버려서 리프가 열려버린다.
                    if self.ref is not None and self.args.ref_adapt > 0:
                        idle = out["mask"] == 0
                        a = self.args.ref_adapt
                        self.ref[idle] = (1.0 - a) * self.ref[idle] + a * tg[idle]
                    # 개방 거리와 액츄에이터 목표는 solve_banks() 가 이미
                    # 두 뱅크 몫으로 이어 붙여 돌려줬다. solve() 가 주는 targets
                    # 는 이미 to_actuator() 를 거친 값이라, 여기서 다시 뒤집으면
                    # 극성이 두 번 뒤집혀 원위치가 되므로 손대지 말 것.
                # 위 두 갈래 중 어느 쪽도 결과를 못 냈을 때만 닫는다.
                # (2026-08-31 버그) 여기가 그냥 else 였다. 광카메라 갈래가
                # 성공해도 열화상 갈래의 else 가 걸려 out 을 도로 지워버렸다.
                # 그래서 source 는 optical 인데 flags 가 없고 리프도 안 열렸다.
                elif out is None:
                    open_mm = [0.0] * self.n
                    act = [leaves.to_actuator(0.0, ch=i + 1)
                           for i in range(self.n)]
                    self.outs = None          # 근거가 없으니 열 상태도 지운다

                if self.manual is not None:
                    open_mm = list(self.manual)
                    # 채널 번호를 같이 넘겨 날 길이 편차(4·8번 5mm)를 반영한다.
                    act = [leaves.to_actuator(v, ch=i + 1)
                           for i, v in enumerate(open_mm)]
                # 불감대는 **카메라가 몰 때만** 크게 잡는다. 열화상 잡음 때문에
                # 목표가 잘게 떨려서 모터가 쉬지 않는 것을 막으려는 것이므로,
                # 사람이 버튼으로 지정한 값에는 걸 이유가 없다. 수동에서는
                # 펌웨어가 무시하기 시작하는 0.3mm 까지 그대로 내보낸다.
                db = MANUAL_DEADBAND_MM if self.manual is not None else self.args.deadband
                due = (now - t_send >= SEND_INTERVAL_S
                       and self.changed(act, db)) or (now - self.link.last_send >= KEEPALIVE_S)
                if self.motor and self.link.ok and due:
                    self.link.send(act)
                    self.sent = list(act)
                    t_send = now
                if self.link.ok and now - t_stat >= STAT_INTERVAL_S:
                    self.link.poll()
                    t_stat = now

                # 주기 FFC 는 배경 스레드(ffc.Periodic)가 돌린다. 여기서 직접 부르면
                # 셔터를 여닫는 2~3초 동안 이 루프가 멈추고, 아두이노 워치독(3초)이
                # 발동해 리프가 강제로 닫힌다.

                # ---- 미리보기 ----
                # **리프 계산에 쓰는 그림(tg)을 그대로 보낸다.** 원본 열화상(temp)을
                # 보내면 화면과 리프가 다른 좌표계를 보게 된다 — 조사야 격자는 4점
                # 보정을 거쳐 펴진 영상이라 원본과 화각·방향이 다르다. 2026-08-30 에
                # 화면상 비어 있는 1번 칸의 리프가 활짝 열려 있는 것으로 드러났다.
                # tg 를 보내면 화면 4칸이 리프 4장과 정확히 같은 영역을 본다.
                if (temp is not None and self.args.preview
                        and now - t_preview >= 1.0 / PREVIEW_HZ):
                    # 기준을 뺀 뒤라면 **뺀 그림**을 보낸다. 화면과 판정이 같은
                    # 것을 봐야 사람이 상태를 옳게 읽는다.
                    # **전체 화면**을 보낸다. 리프가 쓰는 구역은 흰 사각형으로 표시된다.
                    # ⚠️ outs[0] 은 **뒤집힌 그림**으로 푼 결과다 (solve_banks 에서
                    # 솔버 0 이 np.flipud(grid) 를 받는다 — 앞행이 화면 아래쪽이라).
                    # 그래서 마스크도 뒤집힌 좌표계이고, 격자 그림 위에 그대로
                    # 그리면 인체 외곽선이 위아래 반대로 나온다. 되돌려서 그린다.
                    _mask = np.flipud(out["mask"]).copy() if out is not None else None
                    if self.args.preview_grid and tg is not None:
                        # 격자 그림. 화면 8칸 = 리프 8장, 비율도 개구부와 같다.
                        preview_b64 = self.encode_preview_grid(tg, _mask)
                    else:
                        preview_b64 = self.encode_preview(temp, _mask)
                    t_preview = now

                # ---- 상태 뿌리기 ----
                if now - t_state >= 1.0 / STATE_HZ:
                    self.hub.broadcast(self.state(now, temp, tg, out, open_mm, act,
                                                  preview_b64))
                    preview_b64 = None      # 같은 그림을 두 번 보내지 않는다
                    t_state = now

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("\n종료 요청 — 리프를 차폐 위치로 되돌립니다.")
        finally:
            try:
                self.link.close(shield=True)
            except Exception:
                pass
            try:
                self.therm.close()
            except Exception:
                pass

    def set_tmin(self, v):
        """인체 판정 임계 온도를 두 뱅크 솔버에 똑같이 건다.

        한쪽에만 넣으면 같은 사람을 두고 상단은 사람이라 하고 하단은 배경이라
        하는 상태가 되어, 아래쪽 리프만 환자를 물게 된다.
        """
        for s in self.solvers:
            s.tmin = v

    # 중심선 기준일 때 「중심에 인체가 있다」고 볼 최소 온도차 [℃].
    # 중심 온도가 화면 하위 5 % 보다 이만큼도 안 높으면 중심에 사람이 없는
    # 것이므로, 아무것도 인체로 보지 않는다 (= 전 열 SHUT, 실패안전 방향).
    CENTER_MIN_CONTRAST_C = 1.5
    CENTER_WIN = 7          # 중심 온도를 재는 창 (px, 홀수). 한 점은 잡음이 크다.

    def center_temp(self, tg):
        """조사야 격자 정중앙 7×7 의 중앙값 [℃]. 이 자리가 X선 중심선이다."""
        h, w = tg.shape[:2]
        r = self.CENTER_WIN // 2
        cy, cx = h // 2, w // 2
        return float(np.median(tg[cy - r:cy + r + 1, cx - r:cx + r + 1]))

    def body_threshold(self, tg):
        """이 프레임의 인체 판정 하한 온도. None 이면 고정값(s.tmin)을 그대로 쓴다.

        두 방식이 있고 **중심선 기준이 켜져 있으면 그것이 우선**한다.

        · 중심선 기준 (--tmin-center D / 소켓 tmin_center)
            방사선사가 환자를 중심선에 세우므로 그 자리는 항상 인체다. 거기
            온도에서 D 만큼 내린 값을 하한으로 쓴다. 인체가 화면을 얼마나
            차지하든 무관하다 — 백분위 방식은 몸통을 크게 잡으면 상위 16 %
            가 몸 **안쪽**에 떨어져 몸 일부를 잘라냈다 (2026-09-10 관측).
            ⚠️ 중심이 배경(사람 없음)이면 하한이 배경 아래로 내려가 화면
            전체가 인체가 되어 조사야가 활짝 열린다. 그래서 중심이 화면
            하위 5 % 보다 CENTER_MIN_CONTRAST_C 이상 뜨겁지 않으면 「인체
            없음」으로 보고 최고 온도보다 높은 하한을 돌려준다 (전부 차폐).

        · 백분위 기준 (--tmin-pct P)
            상위 (100-P)% 를 인체로. 절대 온도 드리프트에 강하다.
        """
        d = getattr(self.args, "tmin_center", None)
        if d is not None:
            c = self.center_temp(tg)
            lo = float(np.percentile(tg, 5))
            self.last_center = c
            if c - lo < self.CENTER_MIN_CONTRAST_C:
                self.center_ok = False
                return float(tg.max()) + 1.0
            self.center_ok = True
            return c - float(d)
        if self.args.tmin_pct is not None:
            return float(np.percentile(tg, self.args.tmin_pct))
        return None

    def solve_banks(self, grid, mask_ready):
        """두 뱅크를 한 번에 푼다.

        하단 뱅크는 조사야를 **아래에서 위로** 좁힌다. 화면을 위아래로 뒤집으면
        상단 뱅크와 똑같은 문제가 되므로, 뒤집은 그림을 같은 솔버에 한 번 더
        넣는다 (run_row1.py 의 --bank bottom 이 쓰던 처리와 같다).
        위아래만 뒤집으므로 열(좌우) 순서는 그대로다 — 채널 매핑을 바꿀 필요가 없다.

        mask_ready 가 True 면 이미 만들어진 인체 마스크(광카메라 분할), False 면
        온도 그림이다. 두 경우 모두 뒤집기는 똑같이 걸린다.

        반환: (뱅크별 결과 목록, 개방거리 [mm] 8개, 액츄에이터 목표 [mm] 8개)
        """
        outs, open_mm, act = [], [], []
        for i, s in enumerate(self.solvers):
            # ⚠️ 2026-09-07: **솔버 0(ch1~4)이 화면 아래쪽**이다. 실물이 그렇다
            # (윗줄이 5~8번). 그래서 뒤집는 쪽이 0번이다. 예전에는 반대로 걸려
            # 있어서 AUTO 가 엉뚱한 행의 리프를 움직였다.
            g = np.flipud(grid).copy() if i == 0 else grid
            o = s.solve_mask(g) if mask_ready else s.solve(g)
            outs.append(o)
            open_mm += list(s.smoothed)
            act += list(o["targets"])
        return outs, open_mm, act

    def changed(self, act, deadband=None):
        prev = getattr(self, "sent", None)
        if prev is None:
            return True
        db = self.args.deadband if deadband is None else deadband
        return any(abs(a - b) >= db for a, b in zip(act, prev))

    def encode_optical_preview(self):
        """광카메라 사진 위에 표시를 얹어 GUI 로 보낼 그림을 만든다.

        마스크가 광카메라 화면과 같은 좌표라서 외곽선은 그대로 그리면 된다.
        흰 사각형(조사야)만 격자에서 되돌려 그린다.
        """
        img = self.optical_bgr.copy()
        h, w = img.shape[:2]

        if self.outline is not None and self.outline.shape[:2] == (h, w):
            cnts, _ = cv2.findContours(self.outline, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

        M = getattr(self.omap, "M", None)
        if M is not None:
            gh, gw = field.grid_shape()
            corners = np.float32([[0, 0], [gw - 1, 0],
                                  [gw - 1, gh - 1], [0, gh - 1]]).reshape(-1, 1, 2)
            try:
                pts = cv2.perspectiveTransform(corners, np.linalg.inv(M))
                cv2.polylines(img, [pts.astype(np.int32)], True, (255, 255, 255), 2)
            except np.linalg.LinAlgError:
                pass

        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return None
        return base64.b64encode(jpg.tobytes()).decode("ascii")

    def encode_preview_grid(self, tg, mask=None):
        """조사야 격자로 편 그림을 그대로 보낸다 (--preview-grid).

        화면과 리프가 **같은 좌표계**를 본다 — 화면 8칸이 리프 8장과 정확히
        같은 영역이다. 카메라를 90/270도 돌려 쓰면 원본은 세로로 길어지지만
        격자는 개구부 비율(가로가 김)이라 화면 비율도 맞는다.

        ⚠️ 격자 **바깥은 안 보인다.** 정합이 어긋나도 화면에 안 드러나므로,
        보정을 확인할 때는 이 옵션 없이 원본 미리보기를 쓸 것.
        """
        # ⚠️ 위쪽 기준을 98% 로 잡으면 안 된다. 인체가 화면의 15~20% 를
        # 차지하므로 상위 2% 는 **이미 인체 한가운데**다. 그러면 인체 전체가
        # 흰색으로 뭉개지고 배경이 주황까지 밀려 올라가 화면이 통째로 주황이
        # 된다 (2026-09-07 실제로 그랬다).
        # 실제 최댓값을 위 기준으로 쓰면 배경이 아래쪽(어두운 색)으로 내려가
        # 인체와의 온도 차이가 색으로 드러난다.
        # 색을 **인체 판정 경계에서 끊는다.** 균등하게 펴면 배경도 중간 색을
        # 받아 인체와 구분이 안 된다 (2026-09-07: 배경 30.4 / 손 36.2 로 5.8도나
        # 차이가 나는데 화면은 통째로 주황이었다).
        #
        # 임계값 아래 = 어두운 계열, 위 = 밝은 계열로 갈라 놓으면 **기계가
        # 무엇을 인체로 봤는지가 색으로 그대로 드러난다.**
        #
        # ⚠️ 절대 온도는 믿을 수 없다 (같은 장면이 38도로도 8.5도로도 읽힌다).
        # 배경과 인체의 **차이**만 안정적이라, 기준을 백분위로 잡는다.
        #
        # 색 경계는 **판정에 실제로 쓴 하한**과 같아야 한다. 판정은 중심선
        # 기준인데 색은 백분위로 칠하면, 화면이 보여 주는 「인체」와 리프가
        # 움직인 「인체」가 달라 눈으로 검증할 수 없다.
        thr = self.body_threshold(tg)
        if thr is None:
            pct = float(getattr(self.args, "tmin_pct", 84.0) or 84.0)
            thr = float(np.percentile(tg, pct))
        lo = float(np.percentile(tg, 5))
        hi = float(max(tg.max(), thr + 1e-3))
        if thr <= lo:
            thr = lo + 1e-3
        thr = min(thr, hi - 1e-3)       # 「인체 없음」(thr > max) 이면 전부 어둡게
        g = np.interp(np.clip(tg, lo, hi), [lo, thr, hi],
                      [0.0, 90.0, 255.0]).astype(np.uint8)
        img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
        # 마스크는 이미 격자 좌표라 되돌릴 필요가 없다.
        if mask is not None and mask.shape[:2] == tg.shape[:2]:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return None
        return base64.b64encode(jpg.tobytes()).decode("ascii")

    def encode_preview(self, temp, mask=None):
        """사람이 볼 그림을 만든다. **전체 화면**을 보내고 그 위에 표시를 얹는다.

          - 색칠은 **raw 온도**로. 기준을 뺀 영상으로 칠하면 배경이 0 근처라 화면이
            통째로 어두워진다. 계산에는 좋지만 사람이 환자를 보기엔 나쁘다.
          - **흰 사각형** = 리프가 판단에 쓰는 영역 (조사야 격자로 펴는 구역).
            잘라낸 화면만 보내면 환자가 어디 있는지 알 수 없고, 보정이 어긋났을 때
            그 사실조차 안 보인다. 전체를 보여주고 어디를 쓰는지 표시하는 게 낫다.
          - **초록 외곽선** = 기계가 인체로 판단한 영역. 격자 좌표라서 원본으로
            되돌려 그린다.
        """
        # 광카메라 사진이 살아 있으면 그것을 보낸다 (2026-08-31).
        # 열화상은 사람이 보기엔 나쁘다 — 잔상이 남고, 어디가 환자인지 알기 어렵다.
        # 오래된 사진은 안 쓴다. seg_feed 가 죽었는데 옛 그림을 계속 띄우면
        # 화면만 보고는 그 사실을 알 수 없다.
        if (self.optical_bgr is not None
                and time.time() - self.optical_bgr_t < self.args.outline_timeout):
            return self.encode_optical_preview()

        lo = float(np.percentile(temp, 2))
        hi = float(np.percentile(temp, 98))
        g = np.clip((temp - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
        h, w = temp.shape[:2]

        M = getattr(self.tmap, "M", None)
        if M is not None:
            gh, gw = field.grid_shape()

            if mask is not None and mask.shape[:2] == (gh, gw):
                back = cv2.warpPerspective(
                    mask, M, (w, h),
                    flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
                cnts, _ = cv2.findContours(back, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)

            corners = np.float32([[0, 0], [gw - 1, 0],
                                  [gw - 1, gh - 1], [0, gh - 1]]).reshape(-1, 1, 2)
            try:
                pts = cv2.perspectiveTransform(corners, np.linalg.inv(M))
                cv2.polylines(img, [pts.astype(np.int32)], True, (255, 255, 255), 1)
            except np.linalg.LinAlgError:
                pass

        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return None
        return base64.b64encode(jpg.tobytes()).decode("ascii")

    def state(self, now, temp, tg, out, open_mm, act, preview_b64):
        st = {
            "t": round(now, 3),
            "motor": self.motor,
            "manual": self.manual is not None,
            "link": self.link.ok,
            "homed": list(self.link.homed),
            "aligned": self.aligned,
            "n_ch": self.n,
            "travel_mm": self.travel,
            "open_mm": [round(v, 1) for v in open_mm],
            "act_mm": [round(v, 1) for v in act],
            "pos_mm": [None if p is None else round(p, 1) for p in self.link.pos],
            "margin_mm": self.solver.margin,
            "err": self.err or self.link.err,
            "ref": self.ref is not None,
            "source": getattr(self, "source", "thermal"),
            "preparing": self.preparing,
        }
        if self.outs:
            # 열별 상태를 GUI 가 그대로 표시할 수 있는 말로 바꿔 준다.
            #   SHUT  그 열에 인체가 없어 완전 차폐
            #   OVER  인체가 조사야 경계를 넘어감 — 리프를 끝까지 물려도 못 덮는다
            # 상단 4개 다음에 하단 4개가 오도록, 뱅크 순서대로 이어 붙인다.
            flags = []
            for o in self.outs:
                for info in o["infos"]:
                    if info["y_top"] is None:
                        flags.append("SHUT")
                    elif info.get("clamped"):
                        flags.append("OVER")
                    else:
                        flags.append("")
            st["flags"] = flags
            m = self.outs[0]["mask"]        # 마스크는 뒤집기 전후가 같은 그림이다
            st["body_pct"] = round(
                100.0 * float(np.count_nonzero(m)) / m.size, 1)
        if temp is not None:
            st["temp"] = {
                "min": round(float(temp.min()), 1),
                "max": round(float(temp.max()), 1),
                "med": round(float(np.median(temp)), 1),
                "tmin": round(float(self.solver.tmin), 1),
                "tmax": round(float(self.solver.tmax), 1),
            }
            # 중심선 기준이 켜져 있으면 무엇을 봤는지도 같이 보낸다.
            if getattr(self.args, "tmin_center", None) is not None:
                st["temp"]["center_delta"] = float(self.args.tmin_center)
                st["temp"]["center"] = (None if self.last_center is None
                                        else round(self.last_center, 1))
                st["temp"]["center_ok"] = self.center_ok
        if preview_b64:
            st["jpeg"] = preview_b64
        return st


# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thermal", default="/dev/mlc_thermal")
    # 보드 두 대. 채널 1~4 가 상단 뱅크(--port), 5~8 이 하단 뱅크(--port2) 다.
    # udev 이름은 USB 컨트롤러 위치로 붙는 것이라 배선과 무관하다. 리프가
    # 반대로 움직이면 두 값을 맞바꾸면 된다.
    ap.add_argument("--port", default="/dev/mlc_uno2",
                    help="상단 뱅크 아두이노 (리프 1~4)")
    ap.add_argument("--port2", default="/dev/mlc_uno1",
                    help="하단 뱅크 아두이노 (리프 5~8)")
    ap.add_argument("--no-bottom", dest="bottom_bank", action="store_false",
                    help="하단 뱅크 없이 리프 4장만 쓴다")
    ap.add_argument("--port-tcp", type=int, default=PORT, help="GUI 가 붙을 TCP 포트")
    ap.add_argument("--rot", type=int, default=0, choices=(0, 90, 180, 270),
                    help="열화상을 시계방향으로 이만큼 돌려서 쓴다")
    ap.add_argument("--tmin", type=float, default=leaves.TEMP_MIN_C)
    ap.add_argument("--tmax", type=float, default=200.0,
                    help="상한. 백분위 하한을 쓸 때는 넉넉히 열어 둘 것")
    ap.add_argument("--tmin-pct", type=float, default=None,
                    help="상위 (100-N)%% 를 인체로 본다. 센서 드리프트에 강함")
    ap.add_argument("--tmin-center", type=float, default=None, metavar="DELTA",
                    help="조사야 중심(X선 중심선) 온도에서 DELTA ℃ 내린 값을 인체 "
                         "하한으로 쓴다. 환자를 중심에 세우는 프로토콜을 전제. "
                         "켜면 --tmin-pct 보다 우선. 중심이 배경이면 전부 차폐")
    ap.add_argument("--margin", type=float, default=leaves.MARGIN_MM)
    ap.add_argument("--outline-timeout", type=float, default=1.0,
                    help="광카메라 외곽선이 이보다 오래되면 버린다 [s]")
    ap.add_argument("--ref-adapt", type=float, default=0.02,
                    help="기준을 사람 없는 곳만 이 비율로 따라가게 한다 (0=고정)")
    ap.add_argument("--ref-delta", type=float, default=1.5,
                    help="기준 프레임보다 N도 높으면 인체로 본다 (기준을 잡았을 때만)")
    ap.add_argument("--deadband", type=float, default=2.5,
                    help="액츄에이터 목표가 이만큼 안 바뀌면 안 보낸다 [mm]")
    ap.add_argument("--ema", type=float, default=0.25,
                    help="시간 평활 (작을수록 부드럽고 느리다)")
    # 기본 0 = 끔. 카메라가 스스로 약 3분마다 제대로 한다. 2026-08-31 에 우리 FFC 를
    # 완전히 끄고 5분 재보니 중앙값이 0.4도밖에 안 움직였다 — 우리가 걸 이유가 없다.
    ap.add_argument("--ffc-period", type=float, default=0.0,
                    help="우리가 거는 FFC 주기 [s]. 기본 0 = 끔 (카메라에 맡김)")
    ap.add_argument("--no-ffc", action="store_true")
    ap.add_argument("--no-serial", action="store_true", help="모터 없이 판정만")
    ap.add_argument("--no-thermal", action="store_true", help="열화상 없이 소켓만")
    ap.add_argument("--motor-on", action="store_true", help="시작부터 모터 출력 on")
    ap.add_argument("--preview", action="store_true",
                    help="열화상 미리보기 JPEG 를 GUI 로 같이 보낸다")
    ap.add_argument("--rot-image-only", action="store_true",
                    help="회전을 영상에만 적용하고 보정 좌표는 그대로 둔다. "
                         "보정 이후 카메라가 돌아간 경우의 임시 대응 — "
                         "제대로 된 해결은 4점 재보정이다")
    ap.add_argument("--preview-grid", action="store_true",
                    help="미리보기를 조사야 격자로 편 그림으로 보낸다. "
                         "화면 8칸이 리프 8장과 정확히 같은 영역이 되고 "
                         "비율도 개구부와 같아진다. 정합 확인 때는 끄고 쓸 것 "
                         "(격자 바깥이 안 보여 어긋남이 드러나지 않는다)")
    args = ap.parse_args()

    if os.environ.get("VIRTUAL_ENV"):
        print("[경고] 가상환경 안에서 실행 중입니다. Y16(절대 온도)이 막힐 수 있습니다.")
        print("       시스템 python3 로 실행하세요:  /usr/bin/python3 mlc_core.py ...")

    Core(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
