# -*- coding: utf-8 -*-
"""cmd.py — 코어에 명령 한 줄을 보내고 응답을 본다. 진단·수동 조작용.

  python3 cmd.py motor on      # 모터 출력 켜기
  python3 cmd.py motor off     # 끄기
  python3 cmd.py shield        # 전 채널 완전 차폐
  python3 cmd.py auto          # 카메라 판정으로 복귀
  python3 cmd.py home          # 원점 복귀 (약 22초, 로드가 들어갔다 나옵니다)
  python3 cmd.py stop          # 즉시 정지
  python3 cmd.py prepare       # FFC + 기준 프레임 잡기 (환자 들이기 전에)
  python3 cmd.py manual 2 30   # 2번 리프를 30mm 개방
"""

import json
import socket
import sys


def build(argv):
    c = argv[0]
    if c == "motor":
        return {"cmd": "motor", "on": argv[1].lower() in ("on", "1", "true")}
    if c == "manual":
        return {"cmd": "manual", "ch": int(argv[1]), "open_mm": float(argv[2])}
    if c == "margin":
        return {"cmd": "margin", "v": float(argv[1])}
    return {"cmd": c}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    obj = build(sys.argv[1:])
    s = socket.create_connection(("127.0.0.1", 9977), timeout=8)
    f = s.makefile("rwb")
    s.sendall((json.dumps(obj) + "\n").encode("utf-8"))
    print("보냄:", obj)
    while True:
        line = f.readline()
        if not line:
            print("응답 없이 끊김")
            break
        m = json.loads(line)
        if "ok" in m:
            print("응답:", m)
            break
    s.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
