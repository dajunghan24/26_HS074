# -*- coding: utf-8 -*-
"""
optical.py — Raspberry Pi Camera Module 3 (CSI) 취득

열화상은 USB(UVC)라 OpenCV 로 직접 열지만, Camera Module 3 는 CSI 라
libcamera 를 거쳐야 합니다. 라즈베리파이 OS Bookworm 기준으로 python3-picamera2
가 표준입니다. cv2.VideoCapture(0) 으로는 열리지 않는 게 정상입니다.

  sudo apt install -y python3-picamera2

인터페이스는 lepton.LeptonCapture 와 맞춰 두었습니다.

  cam = OpticalCapture()
  bgr = cam.read_bgr()      # (H, W, 3) uint8, OpenCV 순서(BGR)
"""

import threading
import time

import cv2
import numpy as np


class _PiCam2Source:
    """picamera2 백엔드. 라즈베리파이 전용."""

    def __init__(self, size, focus_mode, lens_position):
        from picamera2 import Picamera2          # 여기서만 import (파이 아닌 곳 대비)
        from libcamera import controls

        self.picam2 = Picamera2()
        cfg = self.picam2.create_video_configuration(
            main={"size": (int(size[0]), int(size[1])), "format": "RGB888"})
        self.picam2.configure(cfg)

        # picamera2 의 "RGB888" 은 numpy 배열에서 B,G,R 순으로 나옵니다.
        # 즉 그대로 OpenCV 에 넘기면 색이 맞습니다. 이름에 속지 마세요.
        self.picam2.start()
        time.sleep(0.4)                           # AE/AWB 가 자리잡을 시간

        try:
            if focus_mode == "manual":
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Manual,
                    "LensPosition": float(lens_position),
                })
            elif focus_mode == "auto":
                self.picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
                self.picam2.autofocus_cycle()
            else:                                  # continuous
                self.picam2.set_controls({
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfSpeed": controls.AfSpeedEnum.Fast,
                })
        except Exception as exc:
            # Camera Module 3 가 아니면 AF 컨트롤이 없습니다. 치명적이지 않습니다.
            print("[WARN] 초점 제어를 걸지 못했습니다 (%s). 고정 초점으로 계속합니다." % exc)

        self.name = "picamera2"

    def read(self):
        return self.picam2.capture_array()

    def release(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass


class _CV2Source:
    """USB 웹캠이나 윈도우에서 로직을 확인할 때 쓰는 대체 경로."""

    def __init__(self, index, size):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError("광카메라(index=%s)를 열지 못했습니다." % index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.name = "opencv:%s" % index

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class _Grabber(threading.Thread):
    """열화상과 같은 이유로, 최신 한 장만 들고 있는 스레드."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = source
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            f = self.source.read()
            if f is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = f
                self.seq += 1

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_flag = True


class OpticalCapture:
    """광카메라. picamera2 를 우선 쓰고, 없으면 OpenCV 로 내려간다.

    size 는 '처리에 쓸' 해상도입니다. Camera Module 3 는 4608x2592 까지 되지만
    조사야를 190x230 격자로 펴서 쓸 것이라 640x480 이면 충분하고도 남습니다.
    파이 5 의 CPU 를 리프 계산에 남겨두는 편이 낫습니다.
    """

    def __init__(self, index=None, size=(640, 480),
                 focus_mode="continuous", lens_position=2.0, prefer_cv2=False):
        self.source = None
        errors = []

        if not prefer_cv2:
            try:
                self.source = _PiCam2Source(size, focus_mode, lens_position)
            except Exception as exc:
                errors.append("picamera2: %s" % exc)

        if self.source is None:
            try:
                self.source = _CV2Source(0 if index is None else index, size)
            except Exception as exc:
                errors.append("opencv: %s" % exc)

        if self.source is None:
            raise RuntimeError(
                "광카메라를 열지 못했습니다.\n  " + "\n  ".join(errors) + "\n"
                "라즈베리파이라면:\n"
                "  1) 인식 확인 :  rpicam-hello --list-cameras\n"
                "  2) picamera2 :  sudo apt install -y python3-picamera2\n"
                "     (가상환경을 쓴다면 --system-site-packages 로 만들어야 보입니다)\n"
                "  3) CSI 리본이 파란 면을 올바른 방향으로 꽂혔는지 확인\n"
                "  4) 광카메라 없이 열화상만으로 돌리려면  --fusion off  를 쓰세요\n")

        self.backend = self.source.name
        self.grabber = _Grabber(self.source)
        self.grabber.start()

        # 첫 프레임이 나올 때까지 잠깐 기다립니다.
        for _ in range(60):
            if self.grabber.latest() is not None:
                break
            time.sleep(0.05)
        f = self.grabber.latest()
        self.shape = (0, 0) if f is None else (f.shape[0], f.shape[1])
        print("[OK] 광카메라 — %s, %dx%d" % (self.backend, self.shape[1], self.shape[0]))

    def read_bgr(self):
        return self.grabber.latest()

    def read_gray(self):
        f = self.read_bgr()
        if f is None:
            return None
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f

    def stats(self):
        return self.grabber.seq, 0

    def release(self):
        self.grabber.stop()
        self.grabber.join(timeout=1.0)
        self.source.release()


class MockOptical:
    """카메라 없이 로직만 확인할 때 쓰는 가짜 광학 소스.

    MockThermal 과 같은 위상으로 움직이는 타원을 그려서, 융합 코드가
    '두 영상이 같은 물체를 본다'는 가정 아래 제대로 도는지 볼 수 있게 합니다.
    """

    def __init__(self, w=640, h=480):
        self.w, self.h = w, h
        self.t0 = time.time()
        self.shape = (h, w)
        self.backend = "MOCK"
        self.n = 0

    def read_bgr(self):
        t = time.time() - self.t0
        img = np.full((self.h, self.w, 3), 90, np.uint8)
        img += np.random.normal(0, 3, img.shape).astype(np.int16).clip(-40, 40).astype(np.uint8)

        # MockThermal 의 타원과 같은 비율/위상 (160x120 -> 640x480, 배율 4)
        cy = self.h * 0.5 + 14 * 4 * np.sin(t * 0.7)
        cx = self.w * 0.5 + 30 * 4 * np.sin(t * 0.4)
        cv2.ellipse(img, (int(cx), int(cy)), (22 * 4, 40 * 4), 0, 0, 360, (200, 190, 180), -1)
        self.n += 1
        time.sleep(1.0 / 30.0)
        return img

    def read_gray(self):
        return cv2.cvtColor(self.read_bgr(), cv2.COLOR_BGR2GRAY)

    def stats(self):
        return self.n, 0

    def release(self):
        pass
