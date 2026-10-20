# -*- coding: utf-8 -*-
"""peek.py — 코어 상태를 몇 줄 찍고, 열화상 미리보기를 파일로 저장한다.

진단용 작은 도구입니다. 코어(mlc_core.py)나 가짜 코어(mlc_mock.py)가 떠 있어야 합니다.

  python3 peek.py                    # 3줄 찍고 /tmp/peek.jpg 저장
  python3 peek.py --lines 6 --out /tmp/a.jpg
"""

import argparse
import base64
import json
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9977)
    ap.add_argument("--lines", type=int, default=3)
    ap.add_argument("--gap", type=float, default=1.0, help="줄 사이 간격 [s]")
    ap.add_argument("--out", default="/tmp/peek.jpg")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=8)
    f = sock.makefile("rwb")

    shot = None
    printed = 0
    t_last = 0.0
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        line = f.readline()
        if not line:
            break
        msg = json.loads(line)
        if "ok" in msg:                      # 명령 응답은 건너뛴다
            continue
        if shot is None and "jpeg" in msg:
            shot = msg["jpeg"]
        now = time.time()
        if printed < args.lines and now - t_last >= args.gap:
            t = msg.get("temp") or {}
            print("src=%-8s body=%5.1f%%  ref=%-5s open_mm=%s  flags=%s"
                  % (msg.get("source"), msg.get("body_pct") or 0.0,
                     msg.get("ref"), msg.get("open_mm"), msg.get("flags")))
            printed += 1
            t_last = now
        if printed >= args.lines and shot is not None:
            break

    if shot is not None:
        with open(args.out, "wb") as fp:
            fp.write(base64.b64decode(shot))
        print("그림 저장: %s" % args.out)
    else:
        print("그림 없음 (코어를 --preview 로 띄웠는지 확인하세요)")
    sock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
