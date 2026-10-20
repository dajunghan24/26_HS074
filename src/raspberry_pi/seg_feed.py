# -*- coding: utf-8 -*-
"""seg_feed.py — 광카메라로 사람을 분할해 외곽선을 코어로 보낸다.

  [광카메라] --> MediaPipe 인체 분할 --> 마스크 --소켓--> [코어] --> 리프 구동

2026-08-31 에 방향을 바꾼 결과물이다. 우리에게 필요한 것은 열화상 영상이 아니라
**환자 체형의 외곽선**이고, 그건 광카메라가 훨씬 잘한다:

    해상도   광 640x480  vs  열화상 160x120
    잔상     없음        vs  있음 (하루 종일 씨름했다)
    드리프트 없음        vs  10도까지 흐름
    속도     15ms        vs  9장/초

열화상은 버리지 않는다. '살아있는 사람인가'를 확인하는 보조로 코어에서 계속 돈다.

**mp_env 로 실행할 것** (mediapipe 가 거기에만 있다). 광카메라를 열므로
GUI(thermal_ui2.py)와 동시에 못 쓴다.

  ~/mp_env/bin/python seg_feed.py                 # 화면 없이 코어로 보내기만
  ~/mp_env/bin/python seg_feed.py --show          # VNC 화면에 창을 띄운다
  ~/mp_env/bin/python seg_feed.py --shot /tmp/a.png --once   # 한 장만 저장
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlc_client import MLCClient

MODEL = os.path.expanduser("~/selfie_segmenter.tflite")


class Segmenter:
    """MediaPipe 인체 분할을 감싼다. 사람 마스크(0/255)를 돌려준다."""

    def __init__(self, model_path=MODEL):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        self.seg = vision.ImageSegmenter.create_from_options(opts)

    def mask(self, bgr, min_area_pct=1.0):
        """사람 마스크. 너무 작으면(사람이 없으면) None 을 준다.

        사람이 없을 때 잡음 조각을 사람이라고 보내면 리프가 헛되이 열린다.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.seg.segment(img)
        cat = res.category_mask.numpy_view()
        m = (cat != 0).astype(np.uint8) * 255
        if m.shape[:2] != bgr.shape[:2]:
            m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        pct = 100.0 * float(np.count_nonzero(m)) / m.size
        return (m, pct) if pct >= min_area_pct else (None, pct)


def open_camera(size=(640, 480)):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": size, "format": "RGB888"}))
    cam.start()
    time.sleep(1.0)
    return cam


def draw(bgr, mask, st, pct, fps):
    view = bgr.copy()
    if mask is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cnts, -1, (0, 255, 0), 2)

    lines = ["seg %.1f%%   %.1f fps" % (pct, fps)]
    if st is not None:
        lines.append("open_mm %s" % (st.get("open_mm"),))
        lines.append("source=%s  motor=%s  link=%s"
                     % (st.get("source"), st.get("motor"), st.get("link")))
    else:
        lines.append("core: not connected")
    for i, s in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(view, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9977)
    ap.add_argument("--hz", type=float, default=10.0, help="코어로 보내는 주기")
    ap.add_argument("--cam-hz", type=float, default=5.0, help="광카메라 사진을 코어로 보내는 주기 (GUI 표시용). 0 이면 안 보냄")
    ap.add_argument("--min-area", type=float, default=1.0,
                    help="이보다 작게 잡히면 사람이 없다고 본다 [%%]")
    ap.add_argument("--show", action="store_true", help="VNC 화면에 창을 띄운다")
    ap.add_argument("--shot", default="/tmp/seg_feed.png")
    ap.add_argument("--once", action="store_true", help="한 장만 저장하고 끝")
    ap.add_argument("--no-send", action="store_true", help="코어로 보내지 않는다")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("!! 모델이 없습니다: %s" % args.model)
        return 1

    seg = Segmenter(args.model)
    cam = open_camera()
    client = None
    if not args.no_send:
        client = MLCClient(args.host, args.port)
        client.start()

    win = "MLC — optical + body outline"
    if args.show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 720)
        cv2.moveWindow(win, 0, 0)

    period = 1.0 / max(args.hz, 0.1)
    cam_period = None if args.cam_hz <= 0 else 1.0 / args.cam_hz
    t_send = 0.0
    t_cam = 0.0
    fps, t_prev = 0.0, time.time()
    try:
        while True:
            bgr = cam.capture_array()
            mask, pct = seg.mask(bgr, args.min_area)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt)

            if client is not None and now - t_send >= period:
                # 사람이 없으면 빈 마스크를 보낸다. 그래야 코어가 리프를 닫는다.
                #
                # 사진은 더 뜸하게 보낸다. 리프 판단에는 필요 없고 화면에만 쓰는데,
                # 마스크는 몇 KB 지만 사진은 수십 KB 라 매번 보내면 낭비다.
                send_cam = (cam_period is not None and now - t_cam >= cam_period)
                client.outline(mask if mask is not None
                               else np.zeros(bgr.shape[:2], np.uint8),
                               bgr if send_cam else None)
                if send_cam:
                    t_cam = now
                t_send = now

            st = None if client is None else client.state
            view = draw(bgr, mask, st, pct, fps)

            if args.once:
                cv2.imwrite(args.shot, view)
                print("저장: %s   사람 %.1f%%" % (args.shot, pct))
                break
            if args.show:
                cv2.imshow(win, view)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()
        cam.stop()
        cam.close()
        if client is not None:
            client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
