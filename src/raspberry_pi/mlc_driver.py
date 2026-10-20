# -*- coding: utf-8 -*-
"""
AMLA_4CH_Positioning 호스트 드라이버.

PC(윈도우)에서 벤치 테스트할 때도, 라즈베리파이 5에서 열화상·광카메라 결과로
리프를 구동할 때도 같은 클래스를 씁니다. 포트 이름만 다릅니다.

    from mlc_driver import MLC
    with MLC('COM5') as m:          # 라즈베리파이면 '/dev/ttyUSB0'
        m.home()                    # 원점 복귀 (블로킹)
        m.set([12.0, 30.5, 8.0, 22.0])
        print(m.stat())

단독 실행하면 하드웨어 점검 시퀀스를 돌립니다:

    py mlc_driver.py COM5           # 기본 점검
    py mlc_driver.py COM5 --test    # 배선 자가진단(모터가 하나씩 움직입니다)
"""

import sys
import time

import serial


class MLCError(RuntimeError):
    pass


class MLC:
    """AMLA_4CH_Positioning 펌웨어와 통신하는 클라이언트."""

    N_CH = 4

    def __init__(self, port, baud=115200, timeout=0.3, reset_wait=2.2, verbose=False):
        self.verbose = verbose
        self.ser = serial.Serial(port, baud, timeout=timeout)
        # DTR 로 보드가 리셋되므로 부팅을 기다립니다.
        time.sleep(reset_wait)
        self.ser.reset_input_buffer()

    # ---------- 컨텍스트 매니저 ----------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        try:
            self.ser.write(b"STOP\n")
            time.sleep(0.2)
        except Exception:
            pass
        self.ser.close()

    # ---------- 저수준 ----------

    def _write(self, line):
        if self.verbose:
            print(">", line)
        self.ser.write((line + "\n").encode())

    def send(self, line, expect=("OK", "ERR", "ST"), timeout=3.0):
        """명령을 보내고 OK/ERR/ST 로 시작하는 첫 줄을 돌려줍니다.

        '#' 주석과 'EV' 이벤트는 건너뛰되, 이벤트는 self.events 에 쌓아둡니다.
        """
        self.events = getattr(self, "events", [])
        self.ser.reset_input_buffer()
        self._write(line)

        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "replace").strip()
            if not raw:
                continue
            if self.verbose:
                print("<", raw)
            if raw.startswith("EV"):
                self.events.append(raw)
                continue
            if raw.startswith("#"):
                continue
            if raw.split(" ", 1)[0] in expect:
                if raw.startswith("ERR"):
                    raise MLCError(raw)
                return raw
        raise MLCError("timeout waiting for reply to: %s" % line)

    def drain(self, seconds=0.3):
        """대기 중인 줄을 모두 읽어 리스트로 돌려줍니다."""
        out, deadline = [], time.time() + seconds
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "replace").strip()
            if raw:
                out.append(raw)
        return out

    # ---------- 명령 ----------

    def ping(self):
        return self.send("PING")

    def cfg(self):
        """설정 덤프를 '#' 줄까지 포함해 리스트로 돌려줍니다."""
        self.ser.reset_input_buffer()
        self._write("CFG")
        lines, deadline = [], time.time() + 3.0
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "replace").strip()
            if not raw:
                continue
            lines.append(raw)
            if raw.startswith("OK CFG"):
                break
        return lines

    def stat(self):
        """{'pos': [...], 'busy': [...], 'homed': [...]} 로 파싱해 돌려줍니다."""
        raw = self.send("STAT", expect=("ST",))
        parts = raw.split()
        if len(parts) < 3 + self.N_CH:
            raise MLCError("malformed status: %s" % raw)
        pos = [float(v) for v in parts[1:1 + self.N_CH]]
        busy = [c == "1" for c in parts[1 + self.N_CH]]
        homed = [c == "1" for c in parts[2 + self.N_CH]]
        return {"pos": pos, "busy": busy, "homed": homed}

    def wait_idle(self, timeout=30.0, poll=0.25):
        """전 채널이 멈출 때까지 기다립니다. 마지막 상태를 돌려줍니다."""
        deadline = time.time() + timeout
        st = None
        while time.time() < deadline:
            st = self.stat()
            if not any(st["busy"]):
                return st
            time.sleep(poll)
        raise MLCError("wait_idle timed out; last=%r" % (st,))

    def home(self, ch=None, wait=True, timeout=90.0):
        """원점 복귀. 펌웨어가 채널을 순차로 돌리므로 전 채널은 약 45초 걸립니다."""
        self.send("HOME" if ch is None else "HOME %d" % ch)
        if wait:
            return self.wait_idle(timeout)
        return None

    def go(self, ch, mm, wait=False, timeout=30.0):
        self.send("GO %d %.2f" % (ch, mm))
        if wait:
            return self.wait_idle(timeout)
        return None

    def move(self, ch, dmm, wait=False, timeout=30.0):
        self.send("MOVE %d %.2f" % (ch, dmm))
        if wait:
            return self.wait_idle(timeout)
        return None

    def set(self, targets, wait=False, timeout=30.0):
        """targets: 길이 4 리스트. None 인 항목은 건드리지 않습니다.

        카메라 프레임마다 호출하는 주력 경로입니다.
        """
        toks = ["x" if t is None else "%.2f" % t for t in targets]
        self.send("SET " + " ".join(toks))
        if wait:
            return self.wait_idle(timeout)
        return None

    def stop(self, ch=None):
        return self.send("STOP" if ch is None else "STOP %d" % ch)

    def jog(self, ch, direction, ms):
        """원점 없이 생으로 구동. direction 은 'E'(뻗기) 또는 'R'(들이기)."""
        return self.send("JOG %d %s %d" % (ch, direction, ms), timeout=ms / 1000.0 + 5.0)

    def probe(self, what="MAP", timeout=30.0):
        """A0 프로브 배선 진단 (MAP / GND / VOLT). 펌웨어가 내보내는 줄을 그대로 출력.

        MAP 은 A0 프로브를 L298N 입력 핀 하나에 댄 채로 실행하면, 아두이노가
        핀을 하나씩 흔들어 보고 '그 자리에 몇 번 핀이 물려 있는지' 찾아줍니다.
        선이 끊어져 있으면 OK MAP NONE 이 나옵니다. 배선을 뜯지 않고 단선을
        증명하는 유일한 방법입니다.
        """
        self.ser.reset_input_buffer()
        self._write(what.upper())
        out, deadline = [], time.time() + timeout
        while time.time() < deadline:
            raw = self.ser.readline().decode("utf-8", "replace").strip()
            if not raw:
                continue
            out.append(raw)
            print(raw)
            if raw.startswith("OK ") or raw.startswith("ERR"):
                return out
        raise MLCError("timeout waiting for %s" % what)

    def limit(self, mm):
        return self.send("LIM %.2f" % mm)

    def watchdog(self, on):
        return self.send("WD %d" % (1 if on else 0))

    def failsafe(self, mm):
        return self.send("FS %.2f" % mm)


# =============================== 점검 시퀀스 ===============================

def _p(msg=""):
    print(msg, flush=True)


def checkout(port, run_test=False):
    """하드웨어 점검. 사람이 눈으로 확인해야 하는 부분은 그렇게 안내합니다."""
    _p("=== AMLA 4CH 점검  (%s) ===" % port)

    with MLC(port) as m:
        _p("\n[1] 링크")
        _p("    " + m.ping())

        _p("\n[2] 설정")
        for line in m.cfg():
            _p("    " + line)

        if run_test:
            _p("\n[3] 배선 자가진단 — 액츄에이터를 보세요.")
            _p("    CH1..CH4 를 하나씩 1.5초 뻗었다 1.5초 넣습니다 (약 18초).")
            _p("    ★ 어느 순서에 몇 번 라벨이 움직이는지 적어두세요. ★")
            m.ser.reset_input_buffer()
            m._write("TEST")
            deadline = time.time() + 40
            while time.time() < deadline:
                raw = m.ser.readline().decode("utf-8", "replace").strip()
                if raw:
                    _p("    " + raw)
                if raw.startswith("OK TEST") or raw.startswith("ERR"):
                    break
            _p("\n    자가진단은 개루프입니다. 모터 선이 빠져 있어도 펌웨어는")
            _p("    똑같이 완료를 보고합니다. 실제 동작 여부는 눈으로 본 것이 근거입니다.")
            return

        _p("\n[3] 원점 복귀 (전 채널)")
        t0 = time.time()
        st = m.home(wait=True, timeout=90)
        _p("    %.1f초에 정착   pos=%s  homed=%s"
           % (time.time() - t0, st["pos"], st["homed"]))
        if not all(st["homed"]):
            _p("    !! 원점을 못 잡은 채널이 있습니다. 아래 위치 시험은 건너뜁니다.")
            return

        _p("\n[4] 위치 시험 — 전 채널 20mm 로")
        t0 = time.time()
        m.set([20.0] * MLC.N_CH)
        st = m.wait_idle(timeout=30)
        _p("    %.1f초   pos=%s" % (time.time() - t0, st["pos"]))

        _p("\n[5] 개별 시험 — CH1 만 40mm 로")
        t0 = time.time()
        m.go(1, 40.0)
        st = m.wait_idle(timeout=30)
        _p("    %.1f초   pos=%s" % (time.time() - t0, st["pos"]))

        _p("\n[6] 전 채널 원점 복귀")
        st = m.home(wait=True, timeout=90)
        _p("    pos=%s" % st["pos"])

        _p("\n=== 완료 ===")
        _p("주의: 이 펌웨어는 위치 피드백이 없는 개루프입니다.")
        _p("      위 숫자는 '아두이노가 그렇게 구동했다'는 뜻이지")
        _p("      '리프가 실제로 그 위치에 있다'는 보장이 아닙니다.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    port = args[0] if args else "COM5"
    checkout(port, run_test="--test" in sys.argv)
