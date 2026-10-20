# -*- coding: utf-8 -*-
"""prep.py — 코어에 [준비] 를 보낸다. 방사선사가 환자를 들이기 전에 누르는 그것.

FFC 로 센서를 깨끗하게 만든 뒤, 그 상태의 배경을 기준 프레임으로 잡는다.
화면에 사람이 있으면 코어가 거부한다 (환자를 배경으로 학습하는 사고를 막는다).

  python3 prep.py              # 준비하고 결과를 보여준다
  python3 prep.py --clear      # 기준을 해제한다
"""

import argparse
import json
import socket
import sys
import time


def connect(host, port):
    sock = socket.create_connection((host, port), timeout=8)
    return sock, sock.makefile("rwb")


def cmd(f, obj):
    f.write((json.dumps(obj) + "\n").encode())
    f.flush()
    while True:
        m = json.loads(f.readline())
        if "ok" in m:
            return m


def state(f):
    while True:
        m = json.loads(f.readline())
        if "ok" not in m and "temp" in m:
            return m


def show(st, label):
    t = st.get("temp") or {}
    print("%-10s ref=%-5s preparing=%-5s body=%5.1f%%  tmin=%6.2f  flags=%s"
          % (label, st.get("ref"), st.get("preparing"),
             st.get("body_pct") or 0.0, t.get("tmin", 0.0), st.get("flags")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9977)
    ap.add_argument("--clear", action="store_true", help="기준 해제")
    ap.add_argument("--wait", type=float, default=7.0, help="준비가 끝나기를 기다릴 시간")
    args = ap.parse_args()

    sock, f = connect(args.host, args.port)
    show(state(f), "전")

    if args.clear:
        print("해제 ->", cmd(f, {"cmd": "ref_clear"}))
    else:
        r = cmd(f, {"cmd": "prepare"})
        print("[준비] ->", r)
        if not r.get("ok"):
            sock.close()
            return 1
        print("FFC 걸고 기준 잡는 중... (딸깍 소리가 납니다)")
        time.sleep(args.wait)

    show(state(f), "후")
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
