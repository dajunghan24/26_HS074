#!/bin/bash
# 광카메라 한 장 찍기.  사용법: ./shot.sh /경로/이름.jpg
#
# ⚠️ rpicam-still 은 --mode 2304:1296 과 같이 쓰면 **멈춘다** (2026-09-07 확인).
#    그래서 rpicam-vid 로 mjpeg 을 잠깐 받아 마지막 프레임을 꺼내 쓴다.
# ⚠️ 멈춘 rpicam 을 kill -9 하면 커널 카메라 파이프라인이 깨져 재부팅해야 한다.
#    반드시 일반 kill(TERM) 로 먼저 시도할 것.
# ⚠️ --mode 2304:1296 은 **전체 화각** 모드다. 빼면 1536x864(센서 2/3 크롭,
#    1.5배 확대) 가 잡혀 화각 계산이 통째로 틀어진다.
set -e
OUT="${1:-/tmp/shot.jpg}"
TMP=$(mktemp /tmp/shotXXXX.mjpeg)
rpicam-vid --nopreview --mode 2304:1296 --width 1280 --height 720 \
           --codec mjpeg --framerate 10 -t 2000 -o "$TMP" >/dev/null 2>&1
/usr/bin/python3 -c "
import cv2, sys
cap = cv2.VideoCapture(sys.argv[1]); f = None
while True:
    r, g = cap.read()
    if not r: break
    f = g
cap.release()
if f is None:
    print(\"프레임 없음\"); sys.exit(1)
cv2.imwrite(sys.argv[2], f)
print(\"저장:\", sys.argv[2], f.shape)
" "$TMP" "$OUT"
rm -f "$TMP"
