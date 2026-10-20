# -*- coding: utf-8 -*-
"""ffc.py — Lepton 의 FFC(플랫필드 보정)를 주기적으로 걸어 잔상을 막는다.

Lepton 은 안에 작은 셔터가 있어서, 그걸 닫고 균일한 면을 한 번 봐야 센서의
화소별 편차를 다시 잡는다. 이게 안 돌면 두 가지가 동시에 망가진다.

  1) 이전 장면이 눌어붙어 **잔상**이 남는다
  2) **절대 온도가 통째로 흘러서** 인체 판정 범위(30~41도)가 무의미해진다

## 2026-08-31 정정 — 그전 설명이 틀렸다

예전 주석은 "이 보드에서는 자동 FFC 설정을 V4L2 로 켤 수 없으니 수동으로 대신한다"고
적혀 있었다. **틀렸다.** 설정값을 *읽을* 수 없었을 뿐이고, 카메라의 **자동 FFC 는 원래
잘 돌고 있었다** (약 3분 주기, 셔터가 닫히며 딸깍 소리가 난다).

그런데 그때 대신 쓴 `LEP_CID_RAD_RUN_FFC` 는 **셔터를 닫지 않는** 종류였다. 그래서
카메라가 그 순간의 장면을 균일한 기준면으로 착각하고 보정표에 박아 넣었고, 그 장면이
**검은 잔상**으로 화면에 눌어붙었다. 즉 **잔상을 막으려고 넣은 코드가 잔상을 만들고
있었다.** "처음 받았을 때는 잔상이 없었다"는 사용자 증언이 이걸 가리킨다.

지금은 `run_once()` 가 **셔터를 닫고** FFC 를 건다. 아래 함수 주석을 볼 것.

FFC 동안 약 1초 화면이 멈추고 전체 절차는 2~3초 걸린다. 아두이노 워치독이 3초이므로
**본체 루프가 아니라 배경 스레드(Periodic)에서 돌려야 한다.**
"""

import os
import subprocess
import threading
import time

XML = os.path.expanduser("~/purethermal1-uvc-capture/v4l2/uvcdynctrl/pt1.xml")
CTRL = "LEP_CID_RAD_RUN_FFC"


def device_node(path="/dev/mlc_thermal"):
    """/dev/mlc_thermal 같은 심볼릭 링크를 uvcdynctrl 이 받는 'video8' 로 바꾼다."""
    try:
        return os.path.basename(os.path.realpath(path))
    except Exception:
        return None


def _run(args, timeout=15):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout + p.stderr)
    except Exception as exc:
        return False, str(exc)


def mapped(node):
    ok, out = _run(["uvcdynctrl", "-d", node, "-c"])
    return ok and CTRL in out


def ensure_mapped(node):
    """확장 제어가 등록돼 있는지 보고, 없으면 등록한다.

    등록은 재부팅하면 날아가므로 실행할 때마다 확인한다. sudo 는 비밀번호 없이
    되는 환경에서만 성공하며, 실패해도 프로그램은 그대로 돌아간다(잔상만 남는다).
    """
    if node is None:
        return False
    if mapped(node):
        return True
    if not os.path.exists(XML):
        return False
    _run(["sudo", "-n", "uvcdynctrl", "-d", node, "-i", XML], timeout=30)
    return mapped(node)


SHUTTER = "LEP_CID_SYS_SHUTTER_POSITION"
SHUTTER_CLOSED = "2"
SHUTTER_IDLE = "0"
SETTLE_S = 1.2                 # 셔터가 실제로 움직일 시간


def run_once(node, close_shutter=True):
    """FFC 를 한 번 건다. **반드시 셔터를 닫고** 건다.

    셔터를 안 닫고 `LEP_CID_RAD_RUN_FFC` 만 던지면, 카메라가 **지금 보이는 장면**을
    균일한 기준면으로 착각해 그 모양을 보정표에 박아 넣는다. 그러면 그 장면이 화면에
    **검은 잔상**으로 눌어붙는다. 사람이 있던 자리가 검게 남는 것이 정확히 그것이다.
    2026-08-31 에 실물로 확인했다 — 이 프로젝트가 겪던 잔상의 원인이었다.

    셔터를 닫으면 화면 전체가 같은 온도의 균일한 면이 되므로 보정표가 제대로 만들어진다.
    닫을 때 딸깍 소리가 나고 약 1초간 화면이 멈춘다.

    ⚠️ 전체가 2~3초 걸린다. 아두이노 워치독이 3초이므로 **본체 루프에서 직접 부르지 말고
    배경 스레드(Periodic)에서 부를 것.** 안 그러면 FFC 때마다 리프가 강제로 닫힌다.
    """
    if node is None:
        return False
    if not close_shutter:
        ok, _ = _run(["uvcdynctrl", "-d", node, "-s", CTRL, "1"])
        return ok

    ok_c, _ = _run(["uvcdynctrl", "-d", node, "-s", SHUTTER, SHUTTER_CLOSED])
    time.sleep(SETTLE_S)
    ok_f, _ = _run(["uvcdynctrl", "-d", node, "-s", CTRL, "1"])
    time.sleep(SETTLE_S)
    # 실패하더라도 셔터는 반드시 되돌린다. 닫힌 채로 두면 화면이 아예 안 나온다.
    _run(["uvcdynctrl", "-d", node, "-s", SHUTTER, SHUTTER_IDLE])
    return ok_c and ok_f


class Periodic:
    """period_s 마다 FFC 를 건다. period_s <= 0 이면 아무것도 하지 않는다."""

    def __init__(self, node, period_s):
        self.node = node
        self.period = float(period_s)
        self._stop = threading.Event()
        self._t = None
        self.count = 0

    def start(self):
        if self.period <= 0 or self.node is None:
            return False
        if not ensure_mapped(self.node):
            print("[WARN] FFC 제어를 등록하지 못했습니다. 잔상이 남을 수 있습니다.")
            print("       수동으로:  sudo uvcdynctrl -d %s -i %s" % (self.node, XML))
            return False
        run_once(self.node)                      # 시작할 때 한 번 깨끗하게
        self.count = 1
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        print("[OK] FFC 자동 실행 %.0f 초마다 (%s)" % (self.period, self.node))
        return True

    def _loop(self):
        while not self._stop.wait(self.period):
            if run_once(self.node):
                self.count += 1

    def stop(self):
        self._stop.set()
