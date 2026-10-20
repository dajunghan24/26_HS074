# -*- coding: utf-8 -*-
"""
lepton.py — FLIR Lepton 3.5 + PureThermal(PT3) 라디오메트릭 취득

MLC_final_pi/thermal_mlc.py 에서 '카메라를 여는 부분'만 떼어낸 것입니다.
그 코드는 실제 하드웨어로 검증된 자산이라 로직을 손대지 않았습니다.
바뀐 것은 파일이 나뉜 것과 오류 안내 문구의 옵션 이름뿐입니다.

  from lepton import LeptonCapture, MockThermal, list_cameras, resolve_cam
  cam = LeptonCapture(resolve_cam("/dev/video0"))
  temp_c = cam.read_celsius()      # (120,160) float32, 섭씨
"""

import glob
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

IS_LINUX = sys.platform.startswith("linux")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
#  프레임 정리
# ----------------------------------------------------------------------

def to_raw16(frame):
    """OpenCV가 돌려준 프레임을 (H, W) uint16 배열로 정리한다.

    PureThermal 이 Y16 으로 내보내도 OpenCV 버전/백엔드에 따라
    uint16 2D, uint16 3D, uint8 2채널, uint8 가로 2배 등 여러 모습으로 옵니다.
    RGB 변환이 이미 일어난 8비트 3채널이면 온도 정보가 없으므로 None.
    """
    if frame is None:
        return None
    a = np.asarray(frame)

    if a.dtype == np.uint16:
        if a.ndim == 3:
            a = a[:, :, 0]
        return a

    if a.dtype == np.uint8:
        if a.ndim == 3 and a.shape[2] == 2:
            a = np.ascontiguousarray(a)
            return a.view(np.uint16).reshape(a.shape[0], a.shape[1])
        if a.ndim == 2 and a.shape[1] % 2 == 0 and a.shape[1] >= 320:
            a = np.ascontiguousarray(a)
            return a.view(np.uint16)

    return None  # 8비트 컬러 = 이미 변환됨 -> 온도 못 씀


def strip_telemetry(raw, expect_h=120):
    """PureThermal 이 텔레메트리 행을 붙여 보내면(160x122) 잘라낸다."""
    if raw is not None and raw.shape[0] > expect_h:
        return raw[:expect_h, :]
    return raw


def detect_kelvin_scale(raw):
    """TLinear 해상도가 0.01K 인지 0.1K 인지 판별한다.

    "온도가 그럴듯한가"로 고르면 안 됩니다. 두 후보의 유효 범위가 겹쳐서
    엉뚱한 쪽이 먼저 통과할 수 있습니다(실제로 10배 틀린 값이 나왔습니다).
    실내 온도라면 센티켈빈은 3만 언저리, 데시켈빈은 3천 언저리로
    자릿수가 확실히 다르므로 그걸로 가릅니다.
    """
    med = float(np.median(raw))
    if 20000.0 <= med <= 40000.0:      # 200~400 K = 센티켈빈(0.01)
        return 0.01
    if 2000.0 <= med <= 4000.0:        # 200~400 K = 데시켈빈(0.1)
        return 0.1
    return None


# 라디오메트릭 절대 오차 보정 [°C]. Lepton 3.5 사양 정확도가 +-5도라
# 실제로는 한참 치우쳐 나옵니다. 실행 중 t 키로 잡습니다.
TEMP_OFFSET_C = 0.0
OFFSET_FILE = os.path.join(BASE_DIR, "temp_offset.txt")
REF_PALM_C = 33.0                      # 손바닥 표면 온도 기준값


def load_offset(path=OFFSET_FILE):
    global TEMP_OFFSET_C
    try:
        if os.path.exists(path):
            TEMP_OFFSET_C = float(open(path, encoding="utf-8").read().strip())
            print("[OK] 온도 보정 %+.1f C 적용 (%s)" % (TEMP_OFFSET_C, path))
    except Exception:
        pass
    return TEMP_OFFSET_C


def save_offset(value, path=OFFSET_FILE):
    global TEMP_OFFSET_C
    TEMP_OFFSET_C = float(value)
    with open(path, "w", encoding="utf-8") as f:
        f.write("%.2f" % TEMP_OFFSET_C)


def raw_to_celsius(raw, scale):
    return raw.astype(np.float32) * scale - 273.15 + TEMP_OFFSET_C


# ----------------------------------------------------------------------
#  최신 프레임만 들고 있는 취득 스레드
# ----------------------------------------------------------------------

class _Grabber(threading.Thread):
    """카메라에서 프레임을 쉬지 않고 빨아내어 '가장 최신 것' 한 장만 들고 있는 스레드.

    OpenCV 의 VideoCapture 는 드라이버 쪽에 프레임을 큐로 쌓아둡니다.
    메인 루프가 카메라(8.7fps)보다 조금이라도 느리면 그 큐가 계속 밀려서
    화면이 몇 초 전 장면을 보여줍니다 — 이것이 '잔상'처럼 보이는 지연의 정체입니다.
    여기서 매 프레임을 즉시 버려가며 읽으면 큐가 절대 쌓이지 않습니다.
    """

    def __init__(self, cap, scale, agc=False):
        super().__init__(daemon=True)
        self.cap = cap
        self.scale = scale
        self.agc = agc
        self.lock = threading.Lock()
        self.frame = None
        self.seq = 0
        self.dropped = 0        # 메인 루프가 못 읽고 버린 프레임 수 (지연 진단용)
        self.consumed = 0
        self.stop_flag = False
        self.new_frame = threading.Event()

    def run(self):
        while not self.stop_flag:
            ok, f = self.cap.read()
            if not ok or f is None:
                time.sleep(0.005)
                continue

            if self.agc:
                # 8비트 화면 -> 0~100 상대값. 온도는 아니지만 파이프라인은 동일하게 돈다.
                a = np.asarray(f)
                if a.ndim == 3:
                    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                a = strip_telemetry(a)
                img = a.astype(np.float32) * (100.0 / 255.0)
            else:
                raw = strip_telemetry(to_raw16(f))
                if raw is None:
                    continue
                img = raw_to_celsius(raw, self.scale)

            with self.lock:
                if self.seq > self.consumed:
                    self.dropped += 1        # 이전 프레임을 아무도 안 쓰고 덮어씀
                self.frame = img
                self.seq += 1
            self.new_frame.set()

    def latest(self, timeout=0.3):
        """새 프레임이 올 때까지 잠깐 기다렸다가 최신 한 장을 돌려준다."""
        if not self.new_frame.wait(timeout):
            return None
        with self.lock:
            self.new_frame.clear()
            self.consumed = self.seq
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stop_flag = True


# ----------------------------------------------------------------------
#  백엔드 조합
# ----------------------------------------------------------------------

def _fourcc(a, b, c, d):
    """OpenCV 4 / 5 양쪽에서 동작하는 fourcc 생성."""
    for fn in (getattr(cv2, "VideoWriter_fourcc", None),
               getattr(getattr(cv2, "VideoWriter", None), "fourcc", None)):
        if fn is not None:
            try:
                return fn(a, b, c, d)
            except Exception:
                pass
    return (ord(a) | (ord(b) << 8) | (ord(c) << 16) | (ord(d) << 24))


def _cap_backend(name, default=None):
    """OpenCV 빌드에 없는 백엔드 상수를 안전하게 가져온다."""
    return getattr(cv2, name, cv2.CAP_ANY if default is None else default)


def _capture_attempts():
    """(백엔드 이름, 백엔드, fourcc 쓸지, 강제 해상도) 조합을 OS 에 맞게 만든다.

    라즈베리파이/리눅스에서 UVC 카메라를 여는 백엔드는 V4L2 하나뿐입니다.
    V4L2 로 Y16 을 받으려면 CONVERT_RGB=0 과 fourcc Y16 을 둘 다 걸어야 하고,
    커널이 세로를 122(텔레메트리 포함)로 잡는 경우가 있어 그 조합도 시도합니다.
    """
    if IS_LINUX:
        v4l2 = _cap_backend("CAP_V4L2")
        return [
            ("V4L2", v4l2,        True,  None),
            ("V4L2", v4l2,        True,  (160, 122)),
            ("V4L2", v4l2,        True,  (160, 120)),
            ("V4L2", v4l2,        False, None),
            ("ANY",  cv2.CAP_ANY, True,  None),
            ("ANY",  cv2.CAP_ANY, False, None),
        ]
    dshow, msmf = _cap_backend("CAP_DSHOW"), _cap_backend("CAP_MSMF")
    return [
        ("DSHOW", dshow,       True,  None),
        ("DSHOW", dshow,       True,  (160, 122)),
        ("DSHOW", dshow,       True,  (160, 120)),
        ("DSHOW", dshow,       False, None),
        ("MSMF",  msmf,        True,  None),
        ("MSMF",  msmf,        False, None),
        ("MSMF",  msmf,        True,  (160, 122)),
        ("ANY",   cv2.CAP_ANY, True,  None),
        ("ANY",   cv2.CAP_ANY, False, None),
    ]


def _agc_backends():
    """Y16 이 안 될 때 8비트로라도 열어보는 백엔드 목록."""
    if IS_LINUX:
        return (("V4L2", _cap_backend("CAP_V4L2")), ("ANY", cv2.CAP_ANY))
    return (("DSHOW", _cap_backend("CAP_DSHOW")),
            ("MSMF", _cap_backend("CAP_MSMF")),
            ("ANY", cv2.CAP_ANY))


class LeptonCapture:
    """Y16(라디오메트릭) 취득을 여러 조합으로 시도한다.

    OpenCV 버전과 백엔드 조합에 따라 CONVERT_RGB=0 이 먹기도 하고 무시되기도 해서,
    한 가지 방법만 쓰면 쉽게 실패합니다. 아래 조합을 차례로 시도하고 결과를 다 보여줍니다.
    전부 실패하면 8비트 AGC 모드로 내려가 실험을 계속할 수 있게 합니다.
    """

    ATTEMPTS = _capture_attempts()

    def __init__(self, index, allow_agc=True, force_agc=False, force_scale=None):
        self.force_scale = force_scale
        self.cap = None
        self.backend = None
        self.scale = None
        self.shape = None
        self.grabber = None
        self.agc = False
        self._open(index, allow_agc, force_agc)
        self.grabber = _Grabber(self.cap, self.scale, agc=self.agc)
        self.grabber.start()

    @staticmethod
    def _configure(cap, use_fourcc, size):
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)      # 원본 16비트 유지
        if use_fourcc:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, _fourcc("Y", "1", "6", " "))
            except Exception:
                pass
        if size:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)      # 위 설정으로 풀리는 경우 대비

    @staticmethod
    def _probe(cap, force_scale=None):
        """열린 캡처에서 프레임을 읽어 (raw16, 설명) 을 돌려준다.

        열자마자 나오는 첫 프레임들은 0이거나 반쯤 찬 쓰레기인 경우가 많아서,
        그걸로 스케일을 정하면 10배 틀린 값으로 굳어버립니다.
        앞쪽을 넉넉히 버리고 여러 장의 중앙값으로 판정합니다.
        """
        desc = "프레임 없음"
        got = []
        for i in range(30):
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            a = np.asarray(frame)
            raw = strip_telemetry(to_raw16(frame))
            if raw is None:
                return None, "%s %s (8비트로 변환됨)" % (a.dtype, a.shape)
            if i < 8:
                continue                      # 예열용으로 버림
            got.append(raw)
            if len(got) >= 4:
                break
        if not got:
            return None, desc

        meds = sorted(float(np.median(r)) for r in got)
        med = meds[len(meds) // 2]
        scale = force_scale or detect_kelvin_scale(np.full((1, 1), med, np.float32))
        if scale is None:
            return None, "16비트지만 값 범위 이상 (중앙값 %d)" % int(med)
        raw = got[-1]
        t = med * scale - 273.15
        return (raw, scale), "%s %s  스케일 %.2f K  (중앙값 %d = %.1f C)" % (
            raw.dtype, raw.shape, scale, int(med), t)

    def _open(self, index, allow_agc, force_agc):
        print("Lepton 취득 방법을 순서대로 시도합니다...")
        report = []

        if not force_agc:
            for name, backend, use_fourcc, size in self.ATTEMPTS:
                label = "%-5s fourcc=%-3s size=%s" % (
                    name, "Y16" if use_fourcc else "-",
                    ("%dx%d" % size) if size else "auto")
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    report.append((label, "열지 못함"))
                    cap.release()
                    continue
                self._configure(cap, use_fourcc, size)
                got, desc = self._probe(cap, self.force_scale)
                report.append((label, desc))
                if got is not None:
                    raw, scale = got
                    self.cap, self.backend = cap, name
                    self.scale, self.shape = scale, raw.shape
                    for lab, d in report:
                        print("   %s  ->  %s" % (lab, d))
                    print()
                    print("[OK] Y16 라디오메트릭 취득 성공 — %s, %dx%d, 스케일 %.2f K/count"
                          % (name, raw.shape[1], raw.shape[0], scale))
                    return
                cap.release()

            for lab, d in report:
                print("   %s  ->  %s" % (lab, d))
            print()

        if not allow_agc:
            raise RuntimeError(
                "Lepton 에서 16비트 온도 프레임을 받지 못했습니다.\n"
                "  1) 이 장치가 Y16 을 내보내는지 확인 :\n"
                "       v4l2-ctl -d /dev/video0 --list-formats\n"
                "     목록에 Y16 이 없으면 다른 /dev/videoN 을 --thermal 로 지정하세요.\n"
                "  2) OpenCV 가 5.x 면 Y16 취득이 막힙니다. 4.x 로 맞추세요.\n"
                "  3) 다른 프로그램이 카메라를 잡고 있는지 :  sudo fuser -v /dev/video0\n"
                "  4) 그래도 안 되면 --agc 로 8비트 모드 실험은 계속할 수 있습니다\n")

        # ---- AGC(8비트) 대체 모드 ----
        for name, backend in _agc_backends():
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            for _ in range(12):
                ok, frame = cap.read()
                if ok and frame is not None:
                    a = np.asarray(frame)
                    h = a.shape[0] - (2 if a.shape[0] > 120 else 0)
                    self.cap, self.backend = cap, name + "/AGC"
                    self.scale, self.shape, self.agc = None, (h, a.shape[1]), True
                    print("=" * 62)
                    print(" [AGC 모드] 온도 대신 상대 밝기로 동작합니다.")
                    print("   절대 온도(30~41도) 판정을 못 쓰므로, 화면에서 가장 뜨거운")
                    print("   덩어리를 인체로 봅니다. 리프 계산과 보정은 그대로 동작합니다.")
                    print("=" * 62)
                    return
                time.sleep(0.05)
            cap.release()

        raise RuntimeError(
            "열화상 카메라를 아예 열지 못했습니다.\n"
            "\n"
            "   ★ 위에 '프레임 없음'만 줄줄이 나왔다면 십중팔구 이것입니다:\n"
            "     이 프로그램이 이미 한 번 떠 있습니다. 카메라는 한 번에 한\n"
            "     프로그램만 쓸 수 있어서, 두 번째 실행은 장치를 열어도\n"
            "     프레임을 못 받습니다. 먼저 뜬 창을 q 로 닫거나:\n"
            "         pkill -f run_row1.py\n"
            "\n"
            "   그래도 안 되면 순서대로:\n"
            "   1) 누가 쓰고 있는지 :  sudo fuser -v /dev/video0\n"
            "   2) 장치가 보이는지  :  ls /dev/video*   또는  --list\n"
            "   3) 권한             :  id -nG | grep video\n"
            "      없으면  sudo usermod -aG video $USER  후 재부팅\n"
            "   4) USB 케이블을 뽑았다가 3초 뒤 다시 꽂기\n"
            "      (파이 5 는 파란 USB3 포트보다 검은 USB2 포트가 더 안정적입니다)\n")

    def read_celsius(self):
        return self.grabber.latest()

    def stats(self):
        return self.grabber.seq, self.grabber.dropped

    def release(self):
        if self.grabber is not None:
            self.grabber.stop()
            self.grabber.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class MockThermal:
    """카메라 없이 로직만 확인할 때 쓰는 가짜 열화상 소스."""

    def __init__(self, w=160, h=120):
        self.w, self.h = w, h
        self.t0 = time.time()
        self.shape = (h, w)
        self.backend = "MOCK"
        self.scale = 0.01
        self.agc = False
        self.n = 0

    def read_celsius(self):
        t = time.time() - self.t0
        img = np.full((self.h, self.w), 22.0, np.float32)
        img += np.random.normal(0, 0.25, img.shape).astype(np.float32)

        cy = self.h * 0.5 + 14 * np.sin(t * 0.7)
        cx = self.w * 0.5 + 30 * np.sin(t * 0.4)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        body = (((xx - cx) / 22.0) ** 2 + ((yy - cy) / 40.0) ** 2) < 1.0
        img[body] = 34.0 + np.random.normal(0, 0.3, int(body.sum())).astype(np.float32)
        time.sleep(1.0 / 8.7)
        self.n += 1
        return img

    def stats(self):
        return self.n, 0

    def release(self):
        pass


# ----------------------------------------------------------------------
#  장치 찾기
# ----------------------------------------------------------------------

def _v4l2_device_names():
    """v4l2-ctl 이 깔려 있으면 /dev/videoN -> 장치 이름 표를 만든다."""
    names = {}
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=5).stdout.decode("utf-8", "ignore")
    except Exception:
        return names
    title = ""
    for line in out.splitlines():
        if not line.strip():
            continue
        if line[0] not in " \t":
            title = line.strip().rstrip(":")
        else:
            names[line.strip()] = title
    return names


def _dev_sort_key(path):
    digits = "".join(c for c in path if c.isdigit())
    return int(digits) if digits else 0


def list_cameras(max_index=8):
    """연결된 카메라를 찾아서 보여준다."""
    print("사용 가능한 카메라를 찾는 중...")

    if IS_LINUX:
        names = _v4l2_device_names()
        devs = sorted(glob.glob("/dev/video*"), key=_dev_sort_key)
        if not devs:
            print("  /dev/video* 가 하나도 없습니다.")
            print("  USB 케이블을 다시 꽂고,  dmesg | tail  로 인식 여부를 확인하세요.")
        for dev in devs:
            cap = cv2.VideoCapture(dev, _cap_backend("CAP_V4L2"))
            desc = "열지 못함"
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    a = np.asarray(frame)
                    desc = "%s %s" % (a.dtype, a.shape)
                else:
                    desc = "열렸지만 영상 없음 (메타데이터 노드)"
            cap.release()
            print("  [%-13s] %-38s %s" % (dev, desc, names.get(dev, "")))
        print()
        print("Lepton 은 보통 160x120 또는 세로가 122 로 잡힙니다.")
        print("PureThermal 은 노드를 두 개 만듭니다. 영상이 나오는 쪽(보통 낮은 번호)을 쓰세요.")
        print("  예)  python3 run_row1.py --thermal /dev/video0")
        print()
        print("광카메라(Camera Module 3)는 CSI 라 이 목록으로 판단하지 마세요.")
        print("  확인 :  rpicam-hello --list-cameras")
        return

    for i in range(max_index):
        cap = cv2.VideoCapture(i, _cap_backend("CAP_DSHOW"))
        if cap.isOpened():
            ok, frame = cap.read()
            desc = "읽기 실패"
            if ok and frame is not None:
                a = np.asarray(frame)
                desc = "%s %s" % (a.dtype, a.shape)
            print("  [%d] %s" % (i, desc))
        cap.release()
    print("Lepton 은 보통 160x120 또는 122 로 잡힙니다.")


def resolve_cam(arg):
    """--thermal 값을 VideoCapture 가 받는 형태로 바꾼다.

    숫자면 인덱스(0), 경로면 그대로(/dev/video0) 씁니다.
    라즈베리파이에서는 USB 를 다시 꽂으면 번호가 바뀔 수 있어 경로 쪽이 확실합니다.
    """
    if arg is None:
        return 0
    text = str(arg).strip()
    return int(text) if text.isdigit() else text


# 파이 5 의 CSI/ISP 가 만드는 노드들. 열화상은 여기에 없습니다.
_CSI_HINTS = ("rp1-cfe", "pispbe", "pisp", "unicam", "bcm2835-isp", "rpivid")


def thermal_candidates():
    """열화상일 가능성이 있는 /dev/video* 만 추린다.

    파이 5 는 CSI 카메라와 ISP 때문에 /dev/video0 부터 수십 개의 노드를 만듭니다.
    그래서 USB 열화상이 video0 이라는 보장이 전혀 없고, USB 를 다시 꽂으면
    번호가 통째로 밀리기도 합니다. 이름으로 CSI/ISP 노드를 걸러냅니다.
    """
    if not IS_LINUX:
        return [0]
    names = _v4l2_device_names()
    out = []
    for dev in sorted(glob.glob("/dev/video*"), key=_dev_sort_key):
        title = names.get(dev, "").lower()
        if any(h in title for h in _CSI_HINTS):
            continue
        out.append(dev)
    return out or [0]


def open_thermal(arg=None, allow_agc=True, force_agc=False):
    """열화상을 연다. --thermal 이 있으면 그것만, 없으면 후보를 훑는다."""
    if arg is not None:
        return LeptonCapture(resolve_cam(arg), allow_agc=allow_agc, force_agc=force_agc)

    cands = thermal_candidates()
    print("열화상 후보: %s" % ", ".join(str(c) for c in cands))
    last = None
    for dev in cands:
        print("\n--- %s 시도 ---" % dev)
        try:
            return LeptonCapture(dev, allow_agc=False, force_agc=False)
        except RuntimeError as exc:
            last = exc

    if allow_agc:      # Y16 이 어디서도 안 되면 8비트로라도
        for dev in cands:
            try:
                return LeptonCapture(dev, allow_agc=True, force_agc=True)
            except RuntimeError as exc:
                last = exc

    raise last if last else RuntimeError("열화상 후보가 하나도 없습니다.")
