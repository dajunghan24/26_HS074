# -*- coding: utf-8 -*-
"""seg_test.py — 광카메라 인체 분할이 파이에서 쓸 만한지 잰다.

우리에게 필요한 것은 열화상 영상이 아니라 **환자 체형의 외곽선**이다. 광카메라는
해상도가 4배 높고(640x480 vs 160x120) 잔상도 드리프트도 없으므로, 외곽선을 따는
일에는 광카메라가 낫다. 열화상은 '살아있는 사람인가'를 확인하는 보조로 돌린다.

이 도구는 그 전제를 확인한다:
  - 분할이 사람을 제대로 잘라내는가
  - 파이 5 에서 한 장에 몇 ms 걸리는가 (실시간이 되는가)
  - CPU 를 얼마나 먹는가 (저전압 문제와 직결된다)

**mp_env 로 실행할 것** (mediapipe 가 거기에만 있다):

  ~/mp_env/bin/python seg_test.py --image /tmp/opt.png
  ~/mp_env/bin/python seg_test.py --camera        # 광카메라를 직접 연다 (GUI 를 먼저 끌 것)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

MODEL = os.path.expanduser("~/selfie_segmenter.tflite")


def load_segmenter(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )
    return mp, vision.ImageSegmenter.create_from_options(opts)


def grab_camera(size=(640, 480), warmup=1.0):
    """광카메라에서 한 장. GUI 가 카메라를 잡고 있으면 실패한다."""
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": size, "format": "RGB888"}))
    cam.start()
    time.sleep(warmup)
    frame = cam.capture_array()
    cam.stop()
    cam.close()
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--image", help="저장된 사진으로 시험")
    ap.add_argument("--camera", action="store_true", help="광카메라에서 한 장 찍어 시험")
    ap.add_argument("--runs", type=int, default=10, help="속도를 잴 반복 횟수")
    ap.add_argument("--out", default="/tmp/seg_out.png")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print("!! 모델이 없습니다: %s" % args.model)
        return 1

    if args.camera:
        print("광카메라에서 한 장 찍습니다...")
        bgr = grab_camera()
    elif args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print("!! 사진을 못 읽었습니다: %s" % args.image)
            return 1
    else:
        print("!! --image 나 --camera 중 하나를 주세요")
        return 1

    print("입력: %d x %d" % (bgr.shape[1], bgr.shape[0]))

    t0 = time.time()
    mp, seg = load_segmenter(args.model)
    print("모델 적재: %.2f 초" % (time.time() - t0))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    times = []
    result = None
    for _ in range(args.runs):
        t = time.time()
        result = seg.segment(image)
        times.append((time.time() - t) * 1000.0)

    times = sorted(times)
    print("추론 시간: 중앙값 %.1f ms  (최소 %.1f  최대 %.1f)"
          % (times[len(times) // 2], times[0], times[-1]))
    print("           -> 초당 약 %.1f 장" % (1000.0 / max(times[len(times) // 2], 1e-6)))

    # category_mask: 0 = 배경, 그 외 = 사람
    cat = result.category_mask.numpy_view()
    person = (cat != 0).astype(np.uint8) * 255
    person = cv2.resize(person, (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    pct = 100.0 * float(np.count_nonzero(person)) / person.size
    print("사람으로 잡힌 면적: %.1f%%" % pct)

    view = bgr.copy()
    cnts, _ = cv2.findContours(person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(view, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite(args.out, view)
    print("결과 그림: %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
