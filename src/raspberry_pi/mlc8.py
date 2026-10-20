# -*- coding: utf-8 -*-
"""
mlc8.py — UNO 2대 = 리프 8개를 하나의 번호 체계로 다룬다.

번호 배정 (2026-09-06 확정):

        +-------------------------------------+
    앞행 |  1     2     3     4  |  /dev/mlc_uno1  CH1~CH4
    뒷행 |  5     6     7     8  |  /dev/mlc_uno2  CH1~CH4
        +-------------------------------------+
          왼쪽 ---------------> 오른쪽

사용법
------
    python3 mlc8.py home              8개 전부 완전 수축 (축이 다 들어간다)
    python3 mlc8.py home 5 6 7 8      5~8 번만 수축
    python3 mlc8.py go 30 1 2 3 4     1~4 번을 30mm 로 (같은 보드는 동시 출발)
    python3 mlc8.py go 30             8개 전부 30mm
    python3 mlc8.py solo 30 1         그 보드를 전부 넣은 뒤 1번만 30mm (번호 확인)
    python3 mlc8.py id                1->8 을 하나씩 뻗었다 넣는다 (번호 대조용)
    python3 mlc8.py stat              두 보드의 위치·상태
    python3 mlc8.py setbank /dev/mlc_uno1 R    보드에 자기 행을 각인 (보드당 한 번)

    --together   HOME 을 8채널 겹쳐서 (전원 여유가 확인됐을 때만)

★ 개루프 펌웨어다. 선이 빠져 있어도 똑같이 "완료"를 보고한다.
  판정 근거는 화면 글자가 아니라 눈으로 본 액츄에이터다.
★ 로드가 나올수록 조사야를 가린다. 완전 수축(0mm) = 조사야 완전 개방.
★ 포트를 열 때마다 보드가 리셋돼 원점 기억이 날아간다. 그래서 go 는
  필요하면 알아서 먼저 HOME 을 돌린다.
"""

import sys
import time

from mlc_driver import MLC, MLCError

# 2026-09-06 실물 확인: 포트와 행이 예상과 반대였다.
# /dev/mlc_uno1 에 HOME 을 걸었더니 움직인 것은 '뒷행 왼쪽 첫 번째'(=5번)였다.
# 두 보드는 펌웨어·CFG 가 완전히 같아 소프트웨어로는 구분이 안 되므로,
# 이 표가 유일한 근거다. USB 케이블을 옮겨 꽂으면 이 표도 같이 고쳐야 한다.
BANKS = [
    ("앞행", "/dev/mlc_uno2", [1, 2, 3, 4]),   # 모터 1~4  (사용자 기준 1번 아두이노)
    ("뒷행", "/dev/mlc_uno1", [5, 6, 7, 8]),   # 모터 5~8  (사용자 기준 2번 아두이노)
]

# 뒷행 보드는 모터 극성이 뒤집혀 있었다 (2026-09-06 발견).
# 2026-09-06 펌웨어 AMLA_8CH_75MM 에서 해결했다 — BANK R 인 보드는 IN 두 핀을
# 바꿔 쓴다. 선을 바꿔 끼운 것과 전기적으로 같고, HOME·GO·FS 방향이 한꺼번에
# 맞는다. 그래서 호스트는 이제 아무것도 뒤집지 않는다.
#
# 아래 집합은 '그 뱅크는 아직 극성이 반대다' 라는 표시일 뿐이다 (안내 문구만 바뀐다).
# 펌웨어가 해결했으므로 비어 있다. (앞행=0, 뒷행=1)
INVERTED_BANKS = set()

# 보드가 스스로 신고해야 하는 뱅크 글자. 펌웨어의 EEPROM 에 한 번 저장해 둔 값이다.
# 이 검증이 있으면 포트와 행이 어긋난 채로 도는 사고를 즉시 잡아낸다.
BANK_LETTER = {0: "F", 1: "R"}

HOME_STAGGER_S = 0.4     # 돌입 전류만 어긋내고 구동은 겹친다
HOME_TIMEOUT_S = 45.0    # 한 뱅크(4채널) 기준
MOVE_TIMEOUT_S = 30.0
TRAVEL_MAX_MM = 60.0     # 펌웨어 소프트 상한과 같다
ID_EXTEND_MM = 25.0      # id 시퀀스에서 뻗는 거리


def locate(n):
    """모터 번호 -> (뱅크 인덱스, 보드 안의 채널 1~4)"""
    if not 1 <= n <= 8:
        raise SystemExit("모터 번호는 1~8 입니다: %r" % n)
    return (0, n) if n <= 4 else (1, n - 4)


def group(nums):
    """모터 번호들을 뱅크별로 묶는다 -> {뱅크: [(모터번호, 채널), ...]}"""
    out = {}
    for n in sorted(set(nums)):
        bi, ch = locate(n)
        out.setdefault(bi, []).append((n, ch))
    return out


def open_banks(which=(0, 1)):
    """필요한 뱅크만 연다. 포트를 여는 순간 보드가 리셋되므로 2.2초 기다린다."""
    links = {}
    for bi in sorted(which):
        label, port, nums = BANKS[bi]
        print("[열기] %s  %s  (모터 %s)" % (label, port, "".join(str(x) for x in nums)))
        try:
            links[bi] = MLC(port)
        except Exception as exc:
            close_banks(links)
            raise SystemExit("  포트를 못 엽니다: %s" % exc)
        print("       " + links[bi].ping())
        verify_bank(links, bi)
    return links


def verify_bank(links, bi):
    """보드가 신고한 뱅크가 이 포트에 기대한 행과 맞는지 확인한다.

    두 보드는 겉으로 구분이 안 되므로, 이 확인이 '포트와 행이 맞다'는 유일한
    기계적 근거다. 어긋나면 바로 멈춘다 — 잘못된 행을 구동하는 것보다 낫다.
    """
    want = BANK_LETTER[bi]
    try:
        reply = links[bi].send("BANK")          # "OK BANK R REAR(5-8)"
    except MLCError:
        print("       [주의] 이 보드는 BANK 를 모르는 구버전 펌웨어입니다.")
        print("              포트와 행이 맞는지 확인할 방법이 없습니다.")
        return

    got = ""
    parts = reply.split()
    if len(parts) >= 3:
        got = parts[2]

    if got == want:
        print("       뱅크 확인 %s (%s)" % (got, BANKS[bi][0]))
        return

    if got == "?":
        print("       [주의] 이 보드는 뱅크가 아직 설정되지 않았습니다(UNSET).")
        print("              한 번만 설정하면 EEPROM 에 남습니다:")
        print("              python3 mlc8.py setbank %s %s" % (BANKS[bi][1], want))
        return

    close_banks(links)
    raise SystemExit(
        "\n  !! 뱅크가 어긋났습니다 — 구동을 중단합니다.\n"
        "     %s 로 기대한 것은 '%s'(%s) 인데 보드는 '%s' 라고 신고했습니다.\n"
        "     USB 케이블을 다른 구멍에 옮겨 꽂았거나, BANK 를 잘못 저장한 것입니다.\n"
        "     mlc8.py 의 BANKS 표와 실제 배선을 다시 맞추세요."
        % (BANKS[bi][1], want, BANKS[bi][0], got))


def close_banks(links):
    for m in links.values():
        try:
            m.close()
        except Exception:
            pass


def show(links):
    print("\n     " + "  ".join("%5s" % ("%d번" % n) for n in range(1, 9)))
    row, flags = [], []
    for bi in (0, 1):
        if bi in links:
            st = links[bi].stat()
            row += ["%5.1f" % v for v in st["pos"]]
            flags += ["%s%s" % ("B" if b else ".", "H" if h else "-")
                      for b, h in zip(st["busy"], st["homed"])]
        else:
            row += ["    ?"] * 4
            flags += ["    ?"] * 4
    print("mm   " + "  ".join(row))
    print("상태 " + "  ".join("%5s" % f for f in flags) + "     (B=구동중, H=원점잡음)")


def ensure_homed(links, by_bank, together=False):
    """지정한 채널 중 원점을 안 잡은 것만 HOME 한다. 뱅크 안에서는 0.4초 시차."""
    todo = {}
    for bi, items in by_bank.items():
        st = links[bi].stat()
        need = [(n, ch) for n, ch in items if not st["homed"][ch - 1]]
        if need:
            todo[bi] = need
    if not todo:
        return

    listed = " ".join(str(n) for bi in sorted(todo) for n, _ in todo[bi])
    print("\n  원점을 안 잡은 채널이 있어 먼저 HOME 합니다: %s" % listed)
    t0 = time.time()
    if together:
        for bi in sorted(todo):
            for n, ch in todo[bi]:
                print("    %d번 출발" % n)
                links[bi].send("HOME %d" % ch)
                time.sleep(HOME_STAGGER_S)
        for bi in sorted(todo):
            links[bi].wait_idle(timeout=HOME_TIMEOUT_S)
    else:
        for bi in sorted(todo):
            print("    --- %s ---" % BANKS[bi][0])
            for n, ch in todo[bi]:
                print("    %d번 출발" % n)
                links[bi].send("HOME %d" % ch)
                time.sleep(HOME_STAGGER_S)
            links[bi].wait_idle(timeout=HOME_TIMEOUT_S)
    print("  원점 완료 (%.1f초)" % (time.time() - t0))


def cmd_home(links, nums, together):
    """지정한 모터를 완전 수축시킨다. HOME 은 리미트 스위치까지 확실히 집어넣는다."""
    by_bank = group(nums)
    print("\n=== 원점 복귀(완전 수축) : 모터 %s ==="
          % " ".join(str(n) for n in sorted(set(nums))))
    print("한 채널당 약 10초. 액츄에이터를 보세요.\n")
    t0 = time.time()

    if together:
        for bi in sorted(by_bank):
            for n, ch in by_bank[bi]:
                print("  %d번 출발" % n)
                links[bi].send("HOME %d" % ch)
                time.sleep(HOME_STAGGER_S)
        for bi in sorted(by_bank):
            links[bi].wait_idle(timeout=HOME_TIMEOUT_S)
    else:
        # 기본값: 뱅크(4채널)씩 순서대로. 이미 실측으로 통과한 부하 조건이다.
        for bi in sorted(by_bank):
            print("  --- %s ---" % BANKS[bi][0])
            for n, ch in by_bank[bi]:
                print("  %d번 출발" % n)
                links[bi].send("HOME %d" % ch)
                time.sleep(HOME_STAGGER_S)
            links[bi].wait_idle(timeout=HOME_TIMEOUT_S)
            print("  %s 정착 (%.1f초 경과)" % (BANKS[bi][0], time.time() - t0))

    print("\n%.1f초에 완료." % (time.time() - t0))
    show(links)


def cmd_go(links, mm, nums, together):
    """지정한 모터를 mm 위치로. 같은 보드의 채널은 SET 으로 동시에 출발한다."""
    if not 0.0 <= mm <= TRAVEL_MAX_MM:
        raise SystemExit("행정 범위는 0~%.1fmm 입니다: %r" % (TRAVEL_MAX_MM, mm))
    by_bank = group(nums)

    print("\n=== %.1fmm 로 이동 : 모터 %s ==="
          % (mm, " ".join(str(n) for n in sorted(set(nums)))))
    print("로드가 나올수록 조사야를 가립니다.\n")
    ensure_homed(links, by_bank, together)

    t0 = time.time()
    for bi in sorted(by_bank):
        chans = [ch for _, ch in by_bank[bi]]
        targets = [mm if c in chans else None for c in (1, 2, 3, 4)]
        print("  %s SET %s" % (BANKS[bi][0],
                               " ".join("x" if t is None else "%.1f" % t
                                        for t in targets)))
        links[bi].set(targets)
        links[bi].wait_idle(timeout=MOVE_TIMEOUT_S)
    print("\n%.1f초에 완료." % (time.time() - t0))
    show(links)


def cmd_solo(links, mm, n, together):
    """번호 확인용. 그 모터가 붙은 보드를 전부 수축시킨 뒤 그 하나만 내보낸다.

    한 접속 안에서 끝내야 한다 — 포트를 닫았다 열면 원점 기억이 날아가고,
    그러면 '넣기'와 '내보내기' 사이에 다시 10초짜리 HOME 이 끼어든다.
    """
    bi, ch = locate(n)
    label, _, bank_nums = BANKS[bi]
    other = BANKS[1 - bi][0]
    flipped = bi in INVERTED_BANKS

    print("\n=== %d번 확인 : %s, 왼쪽에서 %d번째 ===" % (n, label, ch))
    if flipped:
        print("!! 이 보드는 극성이 반대다 (배선 미수정). 보이는 방향이 뒤집힌다:")
        print("   먼저 %s 4개(%s)가 모두 '끝까지 나오고',"
              % (label, " ".join(str(x) for x in bank_nums)))
        print("   그 다음 %d번만 %.1fmm '들어갑니다'." % (n, mm))
    else:
        print("먼저 %s 4개(%s)를 모두 넣고, 그 다음 %d번만 %.1fmm 내보냅니다."
              % (label, " ".join(str(x) for x in bank_nums), n, mm))
    print("%s(반대 행)는 건드리지 않습니다 — 지금 자리 그대로 있습니다.\n" % other)

    cmd_home(links, bank_nums, together)
    cmd_go(links, mm, [n], together)

    if flipped:
        print("\n>> 지금 혼자 %.0fmm 들어가 있어야 하는 것은 %s 왼쪽에서 %d번째입니다."
              % (mm, label, ch))
        print(">> (극성이 반대라 '들어가는' 쪽이다. 나머지 3개는 끝까지 나온 채 그대로다.)")
    else:
        print("\n>> 지금 나와 있어야 하는 것은 %s 왼쪽에서 %d번째 하나뿐입니다." % (label, ch))
    print(">> 다른 자리가 움직였다면 그 보드의 OUT 단자 배선이 뒤바뀐 것입니다.")


def cmd_id(links):
    """1번부터 8번까지 하나씩만 움직인다. 라벨과 실물을 대조하는 용도."""
    print("\n=== 번호 대조 : 1번부터 8번까지 하나씩 ===")
    print("각 모터를 %.0fmm 뻗었다가 도로 넣습니다." % ID_EXTEND_MM)
    print("★ 화면에 뜬 번호와 실제로 움직인 리프의 자리가 같은지 보세요.\n")
    for n in range(1, 9):
        bi, ch = locate(n)
        m = links[bi]
        print("  ---- %d번 (%s, 왼쪽에서 %d번째) ----" % (n, BANKS[bi][0], ch))
        m.jog(ch, "E", int(ID_EXTEND_MM * 78))
        time.sleep(1.0)
        m.jog(ch, "R", int(ID_EXTEND_MM * 78) + 800)   # 확실히 되돌린다
        time.sleep(1.0)
    print("\n끝났습니다. 순서가 틀렸다면 그 보드의 OUT 단자 배선을 바꿔야 합니다.")


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    together = "--together" in sys.argv
    if any(a in ("-h", "--help") for a in sys.argv[1:]) or not argv:
        print(__doc__)
        return 0

    action, rest = argv[0], argv[1:]

    if action == "stat":
        links = open_banks()
        try:
            show(links)
        finally:
            close_banks(links)
        return 0

    if action in ("home", "in"):
        nums = [int(x) for x in rest] if rest else list(range(1, 9))
        links = open_banks({locate(n)[0] for n in nums})
        try:
            cmd_home(links, nums, together)
        finally:
            close_banks(links)
        return 0

    if action in ("go", "out"):
        if not rest:
            raise SystemExit("사용법: python3 mlc8.py go <mm> [번호...]")
        mm = float(rest[0])
        nums = [int(x) for x in rest[1:]] if len(rest) > 1 else list(range(1, 9))
        links = open_banks({locate(n)[0] for n in nums})
        try:
            cmd_go(links, mm, nums, together)
        finally:
            close_banks(links)
        return 0

    if action == "solo":
        if len(rest) < 2:
            raise SystemExit("사용법: python3 mlc8.py solo <mm> <번호>")
        mm, n = float(rest[0]), int(rest[1])
        links = open_banks({locate(n)[0]})
        try:
            cmd_solo(links, mm, n, together)
        finally:
            close_banks(links)
        return 0

    if action == "setbank":
        # 포트를 직접 받는다. 뱅크 표를 아직 못 믿는 상황에서 쓰는 명령이기 때문이다.
        if len(rest) < 2:
            raise SystemExit("사용법: python3 mlc8.py setbank /dev/mlc_uno1 R")
        port, letter = rest[0], rest[1].upper()
        if letter not in ("F", "R"):
            raise SystemExit("뱅크는 F(앞행) 또는 R(뒷행) 입니다: %r" % letter)
        print("[열기] %s" % port)
        m = MLC(port)
        try:
            print("       " + m.ping())
            print("       이전: " + m.send("BANK"))
            print("       저장: " + m.send("BANK " + letter))
            print("\n방향이 바뀌었으므로 원점 기억은 버려졌습니다. 다시 HOME 하세요.")
        finally:
            m.close()
        return 0

    if action == "id":
        links = open_banks()
        try:
            cmd_id(links)
        finally:
            close_banks(links)
        return 0

    print(__doc__)
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단됨 — 포트를 닫으며 STOP 을 보냅니다.")
        sys.exit(130)
    except MLCError as exc:
        print("\n[펌웨어 오류] %s" % exc)
        sys.exit(1)
