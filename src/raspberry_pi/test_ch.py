# -*- coding: utf-8 -*-
"""
test_ch.py — 채널 하나만 골라서 시험한다.

전원 부족인지 배선인지 가를 때 씁니다. 한 번에 한 채널만 돌리므로
네 채널을 동시에 돌리는 HOME 과 달리 전류가 1/4 밖에 안 듭니다.

    python3 test_ch.py 1              CH1 을 한 번 (뻗기 2초 -> 들이기 2초)
    python3 test_ch.py 1 5            CH1 을 5번 반복 (됐다 안됐다 확인용)
    python3 test_ch.py 2 3 /dev/ttyUSB0
    python3 test_ch.py 1 1 --home     CH1 만 원점 복귀 (약 11초)

★ 이 펌웨어는 개루프입니다. 선이 빠져 있어도 똑같이 완료를 보고합니다.
  터미널 글자가 아니라 액츄에이터를 보세요.
"""

import sys
import time

from mlc_driver import MLC, MLCError

JOG_MS = 2000          # 한 방향 구동 시간
PAUSE_S = 1.0          # 방향 반전 전 멈춤 (기어 보호)
REST_S = 3.0           # 반복 사이 휴식 (과열 방지)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_home = "--home" in sys.argv

    ch = int(args[0]) if len(args) > 0 else 1
    times = int(args[1]) if len(args) > 1 else 1
    port = args[2] if len(args) > 2 else "/dev/ttyUSB0"

    if not 1 <= ch <= 4:
        raise SystemExit("채널은 1~4 입니다.")

    print("=== CH%d 시험  (%s, %d회) ===" % (ch, port, times))
    print("액츄에이터를 보세요. 화면 글자는 근거가 아닙니다.\n")

    with MLC(port) as m:
        print(m.ping())

        if do_home:
            print("\nCH%d 원점 복귀 (약 11초, 완전 수축)..." % ch)
            st = m.home(ch, wait=True, timeout=45)
            print("   pos=%s  homed=%s" % (st["pos"], st["homed"]))
            return

        ok = 0
        for i in range(times):
            print("--- %d/%d ---" % (i + 1, times))
            try:
                print("   뻗기  %.1f초" % (JOG_MS / 1000.0))
                m.jog(ch, "E", JOG_MS)
                time.sleep(PAUSE_S)
                print("   들이기 %.1f초" % (JOG_MS / 1000.0))
                m.jog(ch, "R", JOG_MS)
                ok += 1
            except MLCError as exc:
                print("   [ERR] %s" % exc)
            if i < times - 1:
                time.sleep(REST_S)

        print("\n명령은 %d/%d 회 전달됐습니다." % (ok, times))
        print("실제로 몇 번 움직였는지는 눈으로 세신 것이 답입니다.")
        print("명령은 다 나갔는데 움직임이 들쭉날쭉하면 → 전원 용량 또는 접촉 불량입니다.")


if __name__ == "__main__":
    main()
