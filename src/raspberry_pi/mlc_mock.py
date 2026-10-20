# -*- coding: utf-8 -*-
"""mlc_mock.py — 팀원 GUI 개발용 가짜 코어.

진짜 코어(mlc_core.py)와 **똑같은 규약**으로 말하지만, 열화상도 아두이노도 쓰지 않습니다.
그래서 파이가 없어도, 장비가 꺼져 있어도, 노트북에서도 GUI 를 만들 수 있습니다.
나중에 이걸 끄고 진짜 코어를 켜면 GUI 는 고칠 것이 없습니다.

  python3 mlc_mock.py               # 127.0.0.1:9977 에서 대기
  python3 mlc_mock.py --port-tcp 9978

가짜로 만들어 내는 것:
  - 리프 4장이 사람이 좌우로 걸어다니는 것처럼 천천히 여닫힘
  - 가끔 SHUT(그 열에 인체 없음) / OVER(조사야 밖까지 걸침) 상태
  - 열화상 미리보기 JPEG (움직이는 뜨거운 덩어리)

명령은 진짜 코어와 같은 것을 받습니다: motor / home / stop / shield / auto /
manual / margin / tmin_pct / tmin / ffc / ping
"""

import argparse
import base64
import json
import math
import socket
import sys
import threading
import time

import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 9977
N_CH = 4
TRAVEL_MM = 57.5
STATE_HZ = 10.0
PREVIEW_HZ = 5.0


def to_actuator(open_mm):
    """개방 거리 -> 액츄에이터 위치. 로드가 나올수록 차폐이므로 뒤집는다."""
    return float(np.clip(TRAVEL_MM - open_mm, 0.0, TRAVEL_MM))


class MockCore:
    def __init__(self, port):
        self.addr = (HOST, port)
        self.clients = []
        self.lock = threading.Lock()
        self.inbox = []
        self.inbox_lock = threading.Lock()

        self.motor = False
        self.manual = None
        self.homed = [False] * N_CH
        self.margin = 5.0
        self.tmin = 28.0
        self.t0 = time.time()

    # ---------- 소켓 ----------

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self.addr)
        srv.listen(5)
        threading.Thread(target=self._accept, args=(srv,), daemon=True).start()
        print("[가짜 코어] %s:%d 에서 GUI 를 기다립니다." % self.addr)
        print("            진짜 장비는 하나도 쓰지 않습니다.")

    def _accept(self, srv):
        while True:
            sock, peer = srv.accept()
            with self.lock:
                self.clients.append(sock)
            print("[가짜 코어] 접속: %s" % peer[0])
            threading.Thread(target=self._read, args=(sock,), daemon=True).start()

    def _read(self, sock):
        buf = b""
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        self._send(sock, {"ok": False, "err": "JSON 아님: %s" % exc})
                        continue
                    with self.inbox_lock:
                        self.inbox.append((msg, sock))
        except OSError:
            pass
        finally:
            with self.lock:
                if sock in self.clients:
                    self.clients.remove(sock)
            sock.close()

    def _send(self, sock, obj):
        try:
            sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        except OSError:
            pass

    def broadcast(self, obj):
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with self.lock:
            targets = list(self.clients)
        for s in targets:
            try:
                s.sendall(data)
            except OSError:
                pass

    # ---------- 명령 ----------

    def handle(self, msg, sock):
        cmd = str(msg.get("cmd", "")).lower()
        try:
            if cmd == "ping":
                pass
            elif cmd == "motor":
                self.motor = bool(msg.get("on", False))
            elif cmd == "home":
                print("[가짜 코어] HOME (진짜는 44초 걸립니다)")
                self.homed = [True] * N_CH
            elif cmd == "stop":
                self.motor = False
            elif cmd == "shield":
                self.manual = [0.0] * N_CH
            elif cmd == "auto":
                self.manual = None
            elif cmd == "manual":
                # 값이 없으면 지금 자리에서 멈춘 채 수동 전환 (AUTO/MANUAL 토글용)
                if self.manual is None:
                    self.manual = list(self.auto_openings())
                if "ch" in msg:
                    i = int(msg["ch"]) - 1
                    if not 0 <= i < N_CH:
                        raise ValueError("ch 는 1~%d" % N_CH)
                    self.manual[i] = float(np.clip(msg.get("open_mm", 0.0),
                                                   0.0, TRAVEL_MM))
                elif "open_mm" in msg:
                    v = msg.get("open_mm")
                    if not isinstance(v, (list, tuple)) or len(v) != N_CH:
                        raise ValueError("open_mm 은 길이 %d 인 배열" % N_CH)
                    self.manual = [float(np.clip(x, 0.0, TRAVEL_MM)) for x in v]
            elif cmd == "margin":
                self.margin = float(msg["v"])
            elif cmd in ("tmin_pct", "tmin"):
                if msg.get("v") is not None:
                    self.tmin = float(msg["v"])
            elif cmd == "ffc":
                print("[가짜 코어] FFC")
            else:
                raise ValueError("모르는 명령: %r" % cmd)
            self._send(sock, {"ok": True, "cmd": cmd})
        except Exception as exc:
            self._send(sock, {"ok": False, "cmd": cmd, "err": str(exc)})

    # ---------- 가짜 데이터 ----------

    def auto_openings(self):
        """사람이 좌우로 천천히 걸어가는 것처럼 리프를 여닫는다."""
        t = time.time() - self.t0
        cx = 1.5 + 1.5 * math.sin(t * 0.35)        # 사람 중심 (열 번호 0~3)
        out = []
        for c in range(N_CH):
            d = abs(c - cx)
            if d > 1.7:
                out.append(0.0)                     # 그 열엔 사람이 없다 -> SHUT
            else:
                out.append(float(np.clip((1.7 - d) * 34.0 + self.margin,
                                         0.0, TRAVEL_MM)))
        return out

    def fake_thermal(self):
        h, w = 120, 160
        t = time.time() - self.t0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = w * (0.5 + 0.28 * math.sin(t * 0.35))
        cy = h * 0.52
        r2 = ((xx - cx) / 26.0) ** 2 + ((yy - cy) / 34.0) ** 2
        temp = 23.0 + 11.0 * np.exp(-r2)
        temp += np.random.normal(0, 0.15, temp.shape)
        return temp

    def preview(self, temp):
        lo, hi = float(np.percentile(temp, 2)), float(np.percentile(temp, 98))
        g = np.clip((temp - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None

    # ---------- 본체 ----------

    def run(self):
        self.start()
        t_state = t_prev = 0.0
        prev_b64 = None
        try:
            while True:
                now = time.time()
                with self.inbox_lock:
                    msgs, self.inbox = self.inbox, []
                for msg, sock in msgs:
                    self.handle(msg, sock)

                if now - t_prev >= 1.0 / PREVIEW_HZ:
                    prev_b64 = self.preview(self.fake_thermal())
                    t_prev = now

                if now - t_state >= 1.0 / STATE_HZ:
                    temp = self.fake_thermal()
                    open_mm = self.manual if self.manual is not None else self.auto_openings()
                    flags = []
                    for v in open_mm:
                        flags.append("SHUT" if v <= 0.01
                                     else ("OVER" if v >= TRAVEL_MM - 0.01 else ""))
                    st = {
                        "t": round(now, 3),
                        "motor": self.motor,
                        "manual": self.manual is not None,
                        "link": True,
                        "homed": list(self.homed),
                        "aligned": True,
                        "n_ch": N_CH,
                        "travel_mm": TRAVEL_MM,
                        "open_mm": [round(v, 1) for v in open_mm],
                        "act_mm": [round(to_actuator(v), 1) for v in open_mm],
                        "pos_mm": [round(to_actuator(v), 1) for v in open_mm],
                        "margin_mm": self.margin,
                        "flags": flags,
                        "body_pct": 18.5,
                        "temp": {"min": round(float(temp.min()), 1),
                                 "max": round(float(temp.max()), 1),
                                 "med": round(float(np.median(temp)), 1),
                                 "tmin": self.tmin, "tmax": 200.0},
                        "err": None,
                        "mock": True,          # 가짜라는 표시. GUI 에 띄워 두면 좋습니다
                    }
                    if prev_b64:
                        st["jpeg"] = prev_b64
                        prev_b64 = None
                    self.broadcast(st)
                    t_state = now

                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n[가짜 코어] 종료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-tcp", type=int, default=PORT)
    args = ap.parse_args()
    MockCore(args.port_tcp).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
