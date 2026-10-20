# -*- coding: utf-8 -*-
"""mlc_client.py — GUI 가 코어에 붙을 때 쓰는 부품.

GUI 쪽 코드에서 소켓·스레드·재접속을 직접 다루지 않아도 되게 감싼 것입니다.
`mp_env` 안에서 돌아갑니다 (numpy 와 cv2 만 씁니다. mediapipe 와 무관).

쓰는 법:

    from mlc_client import MLCClient

    client = MLCClient()
    client.start()                       # 배경에서 알아서 붙고, 끊기면 다시 붙습니다

    st = client.state                    # 최신 상태 dict (아직 없으면 None)
    if st:
        print(st["open_mm"], st["flags"])

    img = client.thermal_bgr()           # 열화상 그림 (numpy BGR) 또는 None

    client.motor(True)                   # 모터 on
    client.nudge(2, +5.0)                # 2번 리프를 5mm 더 연다  <- UP 버튼
    client.nudge(2, -5.0)                #                        <- DOWN 버튼
    client.shield()                      # 전 채널 완전 차폐       <- RESET 버튼
    client.stop_motor()                  # 즉시 정지               <- STOP 버튼

**열화상 장치를 GUI 에서 직접 열지 마세요.** 코어만 엽니다. 그림이 필요하면
`thermal_bgr()` 을 쓰세요 — 코어가 보내 준 것을 디코드해서 돌려줍니다.
"""

import base64
import json
import socket
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:                      # cv2 없이도 상태 수신은 되게
    cv2 = None

HOST = "127.0.0.1"
PORT = 9977
RECONNECT_S = 1.0


class MLCClient:
    """코어와의 연결을 혼자 관리한다. GUI 는 state 만 읽고 명령만 부르면 된다."""

    def __init__(self, host=HOST, port=PORT, on_state=None):
        self.addr = (host, port)
        self.on_state = on_state         # 새 상태가 올 때마다 부를 콜백 (선택)
        self._state = None
        self._jpeg = None                # 마지막으로 받은 JPEG (bytes)
        self._bgr = None                 # 디코드 결과 캐시
        self._bgr_of = None              # 그 캐시가 어느 JPEG 의 것인지
        self._lock = threading.Lock()
        self._sock = None
        self._run = False
        self.connected = False
        self.last_error = None

    # ---------- 수명 ----------

    def start(self):
        if self._run:
            return
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def close(self):
        self._run = False
        self._shutdown()

    def _shutdown(self):
        self.connected = False
        s, self._sock = self._sock, None
        if s is not None:
            try:
                s.close()
            except OSError:
                pass

    # ---------- 배경 스레드 ----------

    def _loop(self):
        while self._run:
            try:
                sock = socket.create_connection(self.addr, timeout=3.0)
                sock.settimeout(None)
                self._sock = sock
                self.connected = True
                self.last_error = None
                self._read_lines(sock)
            except OSError as exc:
                self.last_error = str(exc)
            finally:
                self._shutdown()
            if self._run:
                time.sleep(RECONNECT_S)   # 코어가 다시 뜨면 알아서 붙는다

    def _read_lines(self, sock):
        buf = b""
        while self._run:
            chunk = sock.recv(65536)
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if "ok" in msg:
                    # 명령에 대한 응답. 실패했으면 로그에 남긴다.
                    if not msg.get("ok"):
                        self.last_error = "%s: %s" % (msg.get("cmd"), msg.get("err"))
                        print("[MLC] 명령 거절 — %s" % self.last_error)
                    continue
                self._take(msg)

    def _take(self, msg):
        with self._lock:
            jpg = msg.pop("jpeg", None)
            if jpg:
                try:
                    self._jpeg = base64.b64decode(jpg)
                except Exception:
                    pass
            self._state = msg
        if self.on_state is not None:
            try:
                self.on_state(msg)
            except Exception as exc:
                print("[MLC] on_state 에서 예외: %s" % exc)

    # ---------- 읽기 ----------

    @property
    def state(self):
        with self._lock:
            return self._state

    def thermal_bgr(self):
        """코어가 보내 준 열화상 그림 (numpy BGR). 아직 없으면 None.

        JPEG 는 초당 5번만 옵니다. 그 사이에는 직전 그림을 그대로 돌려줍니다.
        """
        if cv2 is None:
            return None
        with self._lock:
            raw = self._jpeg
        if raw is None:
            return None
        if raw is self._bgr_of:
            return self._bgr
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        self._bgr, self._bgr_of = img, raw
        return img

    def get(self, key, default=None):
        st = self.state
        return default if st is None else st.get(key, default)

    # ---------- 명령 ----------

    def send(self, obj):
        """명령 한 줄. 연결이 없으면 조용히 버린다 (GUI 를 멈추지 않기 위해)."""
        sock = self._sock
        if sock is None:
            return False
        try:
            sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            return True
        except OSError as exc:
            self.last_error = str(exc)
            self._shutdown()
            return False

    def ping(self):
        return self.send({"cmd": "ping"})

    def motor(self, on):
        return self.send({"cmd": "motor", "on": bool(on)})

    def home(self):
        """원점 복귀 후 완전 차폐. **약 45초 걸립니다.** 화면에 진행 표시를 두세요."""
        return self.send({"cmd": "home"})

    def stop_motor(self):
        return self.send({"cmd": "stop"})

    def shield(self):
        """전 채널 완전 차폐."""
        return self.send({"cmd": "shield"})

    def auto(self):
        """카메라 판정으로 복귀 (수동 지정 해제)."""
        return self.send({"cmd": "auto"})

    def manual(self, ch, open_mm):
        """한 채널의 조사야 개방거리를 직접 지정 [mm]."""
        return self.send({"cmd": "manual", "ch": int(ch), "open_mm": float(open_mm)})

    def manual_all(self, open_mm_list):
        return self.send({"cmd": "manual", "open_mm": [float(v) for v in open_mm_list]})

    def nudge(self, ch, delta_mm):
        """지금 값에서 delta 만큼 더 열거나 닫는다. UP / DOWN 버튼용.

        지금 값을 모르면(상태가 아직 안 왔으면) 아무것도 하지 않습니다.
        """
        st = self.state
        if st is None:
            return False
        i = int(ch) - 1
        cur = st.get("open_mm", [])
        if not 0 <= i < len(cur):
            return False
        limit = float(st.get("travel_mm", 57.5))
        want = min(max(float(cur[i]) + float(delta_mm), 0.0), limit)
        return self.manual(ch, want)

    def margin(self, mm):
        return self.send({"cmd": "margin", "v": float(mm)})

    def tmin_pct(self, pct):
        return self.send({"cmd": "tmin_pct", "v": float(pct)})

    def ffc(self):
        return self.send({"cmd": "ffc"})

    def outline(self, mask, bgr=None):
        """광카메라에서 딴 인체 마스크를 코어로 보낸다 (0/255 uint8).

        필요한 것은 열화상 영상이 아니라 **환자 체형의 외곽선**이다. 광카메라 분할은
        640x480 으로 열화상(160x120)보다 4배 정밀하고 잔상·드리프트가 없다.
        그래서 외곽선은 이쪽이 맡고, 열화상은 보조로 돈다.

        PNG 로 압축해 보낸다. 0/255 이진 영상이라 몇 KB 밖에 안 되므로 초당 10장도
        loopback 에서는 부담이 없다.
        """
        if cv2 is None or mask is None:
            return False
        ok, png = cv2.imencode(".png", mask)
        if not ok:
            return False
        obj = {
            "cmd": "outline",
            "w": int(mask.shape[1]), "h": int(mask.shape[0]),
            "png": base64.b64encode(png.tobytes()).decode("ascii"),
        }
        # 광카메라 사진도 같이 보낸다. 카메라는 한 프로그램만 열 수 있어서
        # GUI 가 직접 못 연다 — 이미 쥐고 있는 쪽이 넘겨주는 수밖에 없다.
        if bgr is not None:
            ok, jpg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                obj["cam"] = base64.b64encode(jpg.tobytes()).decode("ascii")
        return self.send(obj)


# 혼자 돌려 보면서 규약을 확인할 때 쓰는 것.
if __name__ == "__main__":
    c = MLCClient()
    c.start()
    print("코어에 붙는 중... (mlc_core.py 나 mlc_mock.py 가 떠 있어야 합니다)")
    try:
        while True:
            time.sleep(1.0)
            st = c.state
            if st is None:
                print("  대기 중 — %s" % (c.last_error or "아직 연결 안 됨"))
                continue
            img = c.thermal_bgr()
            print("  open_mm=%s flags=%s motor=%s 그림=%s"
                  % (st["open_mm"], st.get("flags"), st["motor"],
                     "없음" if img is None else "%dx%d" % (img.shape[1], img.shape[0])))
    except KeyboardInterrupt:
        c.close()
