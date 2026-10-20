from kivy.config import Config

Config.set("graphics", "width", "800")
Config.set("graphics", "height", "480")
Config.set("graphics", "resizable", "0")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line, RoundedRectangle
from kivy.core.text import Label as CoreLabel
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.properties import ColorProperty
from kivy.uix.widget import Widget

import cv2
import numpy as np
import subprocess
import mediapipe as mp
import os
import threading
import time

from mlc_client import MLCClient


Window.size = (800, 480)

# 창 바탕을 흰색으로. 이걸 바꾸면 **글자색도 같이 바꿔야 한다** — Kivy 의
# Label·Button 기본 글자색이 흰색이라 흰 배경에서는 그냥 사라진다.
Window.clearcolor = (1, 1, 1, 1)

# ---- iOS 라이트모드 시스템 색 ------------------------------------------
# 흰 배경 위에서 쓰는 값이다. 다크모드 값(#0A84FF, #30D158 등)은 흰 바탕에서
# 물이 빠져 흐릿해 보인다.
IOS_FILL = (0.898, 0.898, 0.918, 1)   # #E5E5EA  중립 버튼 (UP/DOWN)
IOS_LABEL = (0.11, 0.11, 0.118, 1)    # #1C1C1E  흰 바탕 위 글자
IOS_BLUE = (0.00, 0.478, 1.00, 1)     # #007AFF  기준 자세 복귀 (RESET)
IOS_GREEN = (0.204, 0.780, 0.349, 1)  # #34C759  AUTO
IOS_ORANGE = (1.00, 0.584, 0.00, 1)   # #FF9500  MANUAL
IOS_RED = (1.00, 0.231, 0.188, 1)     # #FF3B30  STOP
IOS_DIM = (0.78, 0.78, 0.80, 1)       # #C7C7CC  연결 없음

# 한글과 ▲▼✖ 를 함께 가진 폰트. Kivy 는 위젯당 폰트가 하나뿐이라 둘 다 필요하다.
# ⚠(U+26A0)·⛔(U+26D4)는 나눔에 없어서 못 쓴다 — 쓰면 □ 로 나온다.
KOREAN_FONT = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"


class RoundedButton(Button):
    """모서리가 둥근 iOS 풍 버튼.

    Kivy 의 `<Button>` 규칙은 `canvas.before` 에 Color + BorderImage 를 깐다.
    `background_normal=""` 로 두면 그림이 사라지는 게 아니라 **배경색으로 꽉 찬
    평평한 사각형**이 된다. 그리고 그 명령들은 `__init__` 에서 지워도 **다음
    프레임에 되살아난다** — 2026-09-07 에 실험으로 확인했다.

    그래서 지우는 대신 **투명하게** 만든다. `background_color` 를 (0,0,0,0) 으로
    고정하면 규칙이 그리는 사각형이 안 보이고, 그 위에 우리가 얹은 둥근 사각형만
    남는다.

    ⚠️ 그 결과 **`background_color` 로는 색이 안 바뀐다.** 색은 `fill` 로 바꿀 것.

    누른 상태는 iOS 처럼 **어두워지기만 하고 모양은 그대로**다. 크기가 변하면
    터치스크린에서 손가락 밑이 흔들리는 것처럼 보인다.
    """

    fill = ColorProperty(IOS_FILL)

    def __init__(self, radius=22, **kwargs):
        # 규칙이 그리는 사각형을 투명하게. 이 값은 바꾸지 말 것 — 바꾸면 각진
        # 사각형이 둥근 것 위로 되돌아온다.
        kwargs["background_normal"] = ""
        kwargs["background_down"] = ""
        kwargs["background_disabled_normal"] = ""
        kwargs["background_color"] = (0, 0, 0, 0)
        kwargs.setdefault("font_name", KOREAN_FONT)
        kwargs.setdefault("markup", True)      # 두 번째 줄만 작게 쓰려고
        kwargs.setdefault("halign", "center")
        kwargs.setdefault("valign", "middle")
        # 글자 크기가 다른 두 줄은 기본 간격(1.0)이면 사이가 벌어져 보인다.
        kwargs.setdefault("line_height", 0.82)
        super(RoundedButton, self).__init__(**kwargs)
        # Kivy Label 의 기본 정렬은 왼쪽이다. text_size 를 폭에 묶지 않으면
        # 두 줄의 길이가 다를 때 짧은 줄이 왼쪽으로 붙는다.
        self.bind(size=lambda w, s: setattr(w, "text_size", (s[0], None)))

        # radius 는 네 모서리 값을 모두 준다. 하나만 주면 나중에 속성으로
        # 대입할 때 무시된다.
        self.corner = [radius] * 4
        with self.canvas.before:
            self._c = Color(*self.fill)
            self._r = RoundedRectangle(pos=self.pos, size=self.size,
                                       radius=self.corner)
        self.bind(pos=self._restyle, size=self._restyle,
                  state=self._restyle, fill=self._restyle)

    def _restyle(self, *_):
        r, g, b, a = self.fill
        if self.state == "down":
            # 밝은 바탕색을 많이 어둡게 하면 '고장난 것' 처럼 보인다.
            # iOS 도 눌린 동안 살짝 가라앉을 뿐이다.
            r, g, b = r * 0.82, g * 0.82, b * 0.82
        self._c.rgba = (r, g, b, a)
        self._r.pos = self.pos
        self._r.size = self.size
        self._r.radius = self.corner




BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "/home/ict/hand_landmarker.task"


class _NoHands:
    """MediaPipe 를 껐을 때 쓰는 빈 결과.

    아래 코드가 result.hand_landmarks 만 보므로 이것으로 충분합니다.
    """
    hand_landmarks = []


_NO_HANDS = _NoHands()

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)



CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ---- 광카메라 정렬 보정 (2026-09-07 실측) ------------------------------
# 카메라가 물리적으로 4.76도 비틀려 달렸고 더는 못 잡는다고 해서 소프트웨어로
# 편다. 부호는 실험으로 정했다 — +4.76 에서 잔여 회전이 0.00 도가 됐다.
# 남은 중심 어긋남도 같이 민다.
#
# ⚠️ 카메라를 물리적으로 다시 만지면 이 값들을 다시 재야 한다.
#    조사야 램프를 켜고 한 프레임 받아 field_measure.py 에 넣으면 나온다.
OPTICAL_ROLL_DEG = 0.14      # 2026-09-07 오후 재설치 후 재측정. 물리적으로 거의 수평
OPTICAL_DX_PX = 16.7         # 640x480 기준 (1280x720 에서 잰 25px x 0.667)
OPTICAL_DY_PX = -32.0        # 640x480 기준 (-48px x 0.667)

# 센서 모드를 전체 화각으로 못박는다. 이걸 안 주면 libcamera 가 640x480 요청에
# 1536x864 모드를 골라 **1.5 배 확대된** 그림을 준다. 그러면 field.py 가 아는
# 화각과 실제가 어긋나 픽셀->mm 변환이 통째로 틀린다.
OPTICAL_SENSOR_MODE = "2304:1296"

# 열화상 장치는 이제 GUI 가 열지 않는다. 코어(mlc_core.py)만 연다.
# /dev/videoN 번호는 USB 를 다시 꽂을 때마다 바뀌므로 박아 두면 안 된다
# (2026-08-30 에 video8 -> video9 로 바뀌는 것을 실제로 겪었다).
MLC_HOST = "127.0.0.1"
MLC_PORT = 9977
# ---- 화면 칸 <-> 모터 번호 --------------------------------------------
# ⚠️ 2026-09-07 실물 확인: **화면 윗줄이 5~8번 모터, 아랫줄이 1~4번**이다.
# (윗줄 = /dev/mlc_uno1 = 뒷행 보드, 아랫줄 = /dev/mlc_uno2 = 앞행 보드)
#
# 화면 칸 번호와 모터 번호가 다르므로 **반드시 이 함수들을 거칠 것.**
# 셋 다 (x+4)%8 로 같지만 이름을 나눠 둔 것은 어느 쪽 번호를 다루는지
# 코드에서 드러나게 하려는 것이다 — 예전에 `sel + 1` 이 곳곳에 흩어져 있었고
# 그게 행이 뒤바뀐 채로 오래 굴러간 원인이었다.


def ch_index(cell):
    """화면 칸(0~7) -> 코어 채널 인덱스(0~7)"""
    return (cell + 4) % 8


def motor_no(cell):
    """화면 칸(0~7) -> 사람이 부르는 모터 번호(1~8)"""
    return ch_index(cell) + 1


def cell_of_ch(idx):
    """코어 채널 인덱스(0~7) -> 화면 칸(0~7). 반쪽 맞바꿈이라 식이 같다."""
    return (idx + 4) % 8


class CollimatorArea(Widget):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)


        # 기준 자세는 **최대 개방**이다 (2026-09-07 RESET 의 뜻이 바뀌었다).
        # 위 행 1.00 = 위쪽 가장자리까지 물러남, 아래 행 0.00 = 아래쪽 끝.
        # 0.50 은 중앙선, 즉 두 행이 만나 완전히 가린 자리다.
        self.original_boundaries = [
            1.00,
            1.00,
            1.00,
            1.00,
            0.00,
            0.00,
            0.00,
            0.00
        ]

        self.boundaries = list(
            self.original_boundaries
        )

        self.selected = None

        self.thermal_frame = None

        self.bind(
            size=self.update_graphics,
            pos=self.update_graphics
        )

        self.update_graphics()


    def set_thermal_frame(self, frame):

        self.thermal_frame = frame

        self.update_graphics()


    def update_graphics(self, *args):

        self.canvas.clear()

        with self.canvas:


            if self.thermal_frame is not None:

                Color(
                    1,
                    1,
                    1,
                    1
                )

                Rectangle(
                    texture=self.thermal_frame,
                    pos=self.pos,
                    size=self.size
                )

            else:

                Color(
                    0.03,
                    0.03,
                    0.03,
                    1
                )

                Rectangle(
                    pos=self.pos,
                    size=self.size
                )

            column_width = (
                self.width / 4
            )


            for column in range(4):

                x = (
                    self.x +
                    column * column_width
                )

                top_index = column

                top_boundary = (
                    self.boundaries[top_index]
                )

                top_y = (
                    self.y +
                    top_boundary *
                    self.height
                )

                top_height = (
                    self.y +
                    self.height -
                    top_y
                )


                if self.selected == top_index:

                    Color(
                        0.8,
                        0.05,
                        0.05,
                        0.55
                    )

                else:

                    Color(
                        0.02,
                        0.02,
                        0.02,
                        0.80
                    )

                Rectangle(
                    pos=(
                        x,
                        top_y
                    ),
                    size=(
                        column_width,
                        max(
                            0,
                            top_height
                        )
                    )
                )


                Color(
                    0.1,
                    0.7,
                    1,
                    1
                )

                Line(
                    points=[
                        x,
                        top_y,
                        x,
                        self.y +
                        self.height
                    ],
                    width=2
                )

                Line(
                    points=[
                        x + column_width,
                        top_y,
                        x + column_width,
                        self.y +
                        self.height
                    ],
                    width=2
                )


                Line(
                    points=[
                        x,
                        self.y +
                        self.height,
                        x + column_width,
                        self.y +
                        self.height
                    ],
                    width=2
                )


                if self.selected == top_index:

                    Color(
                        1,
                        0.1,
                        0.1,
                        1
                    )

                    Line(
                        points=[
                            x,
                            top_y,
                            x + column_width,
                            top_y
                        ],
                        width=5
                    )

                else:

                    Color(
                        0.1,
                        0.7,
                        1,
                        1
                    )

                    Line(
                        points=[
                            x,
                            top_y,
                            x + column_width,
                            top_y
                        ],
                        width=2
                    )


                number_label = CoreLabel(
                    text=str(
                        motor_no(top_index)
                    ),
                    font_size=32,
                    bold=True
                )

                number_label.refresh()

                tw = number_label.texture.width
                th = number_label.texture.height

                Color(
                    1,
                    1,
                    1,
                    1
                )

                Rectangle(
                    texture=number_label.texture,
                    pos=(
                        x +
                        (column_width - tw) / 2,
                        top_y +
                        (top_height - th) / 2
                    ),
                    size=(
                        tw,
                        th
                    )
                )


                bottom_index = column + 4

                bottom_boundary = (
                    self.boundaries[
                        bottom_index
                    ]
                )

                bottom_y = self.y

                bottom_top_y = (
                    self.y +
                    bottom_boundary *
                    self.height
                )

                bottom_height = (
                    bottom_top_y -
                    self.y
                )


                if self.selected == bottom_index:

                    Color(
                        0.8,
                        0.05,
                        0.05,
                        0.55
                    )

                else:

                    Color(
                        0.02,
                        0.02,
                        0.02,
                        0.80
                    )

                Rectangle(
                    pos=(
                        x,
                        bottom_y
                    ),
                    size=(
                        column_width,
                        max(
                            0,
                            bottom_height
                        )
                    )
                )


                Color(
                    0.1,
                    0.7,
                    1,
                    1
                )

                Line(
                    points=[
                        x,
                        self.y,
                        x,
                        bottom_top_y
                    ],
                    width=2
                )

                Line(
                    points=[
                        x + column_width,
                        self.y,
                        x + column_width,
                        bottom_top_y
                    ],
                    width=2
                )


                Line(
                    points=[
                        x,
                        self.y,
                        x + column_width,
                        self.y
                    ],
                    width=2
                )


                if self.selected == bottom_index:

                    Color(
                        1,
                        0.1,
                        0.1,
                        1
                    )

                    Line(
                        points=[
                            x,
                            bottom_top_y,
                            x + column_width,
                            bottom_top_y
                        ],
                        width=5
                    )

                else:

                    Color(
                        0.1,
                        0.7,
                        1,
                        1
                    )

                    Line(
                        points=[
                            x,
                            bottom_top_y,
                            x + column_width,
                            bottom_top_y
                        ],
                        width=2
                    )


                number_label = CoreLabel(
                    text=str(
                        motor_no(bottom_index)
                    ),
                    font_size=32,
                    bold=True
                )

                number_label.refresh()

                bw = number_label.texture.width
                bh = number_label.texture.height

                Color(
                    1,
                    1,
                    1,
                    1
                )

                Rectangle(
                    texture=number_label.texture,
                    pos=(
                        x +
                        (column_width - bw) / 2,
                        self.y +
                        (bottom_height - bh) / 2
                    ),
                    size=(
                        bw,
                        bh
                    )
                )


            Color(
                0.8,
                0.8,
                0.8,
                1
            )

            Line(
                rectangle=(
                    self.x,
                    self.y,
                    self.width,
                    self.height
                ),
                width=2
            )

            # ── X선 중심선(center ray) 표식 ──────────────────────────
            # 코어를 --preview-grid 로 띄우면 이 패널이 곧 조사야라서,
            # 패널 정중앙이 X선 중심선이다. 방사선사가 환자를 세울 때
            # 어디가 중심인지 한눈에 알 수 있게 십자 + 원을 그린다.
            # 리프 사각형 위에 그려야 완전 차폐 상태에서도 보이므로
            # 맨 마지막에 둔다. 2026-09-10 사용자 요청.
            #
            # ⚠️ --preview-grid 없이(원본 전체 화면) 띄우면 패널 중앙이
            #    중심선이 아니다. 그때는 이 표식을 믿지 말 것.
            cx = self.x + self.width / 2.0
            cy = self.y + self.height / 2.0
            arm = self.width * 0.10       # 팔 길이
            gap = self.width * 0.025      # 가운데 비움 (원이 들어갈 자리)
            rad = self.width * 0.018      # 원 반지름

            Color(1.0, 0.82, 0.20, 1)     # 앰버 — 파란 리프선·초록 외곽선과 구분
            Line(points=[cx - arm, cy, cx - gap, cy], width=2)
            Line(points=[cx + gap, cy, cx + arm, cy], width=2)
            Line(points=[cx, cy - arm, cx, cy - gap], width=2)
            Line(points=[cx, cy + gap, cx, cy + arm], width=2)
            Line(circle=(cx, cy, rad), width=1.5)


    def on_touch_down(self, touch):

        if not self.collide_point(
            *touch.pos
        ):
            return False


        touch_x = touch.x
        touch_y = touch.y


        relative_x = (
            touch_x - self.x
        )

        relative_y = (
            touch_y - self.y
        )


        corrected_y = (
            self.height -
            relative_y
        )


        column_width = (
            self.width / 4
        )

        column = int(
            relative_x /
            column_width
        )

        column = max(
            0,
            min(
                3,
                column
            )
        )


        if corrected_y <= (
            self.height * 0.50
        ):

            # TOP = 1~4

            self.selected = column

        else:

            # BOTTOM = 5~8

            self.selected = column + 4


        print(
            "TOUCH:",
            round(touch_x, 1),
            round(touch_y, 1),
            "-> 모터",
            motor_no(self.selected)
        )

        self.update_graphics()

        return True


    def move_selected(
        self,
        direction
    ):

        if self.selected is None:

            print(
                "NO COLLIMATOR SELECTED"
            )

            return

        index = self.selected

        move_amount = 0.03


        if index < 4:

            if direction == "up":

                self.boundaries[index] += (
                    move_amount
                )

            elif direction == "down":

                self.boundaries[index] -= (
                    move_amount
                )


        else:

            if direction == "up":

                self.boundaries[index] += (
                    move_amount
                )

            elif direction == "down":

                self.boundaries[index] -= (
                    move_amount
                )


        self.boundaries[index] = max(
            0.0,
            min(
                1.0,
                self.boundaries[index]
            )
        )

        self.update_graphics()

        print(
            "COLLIMATOR",
            index + 1,
            "MOVE",
            direction
        )


    def reset_collimators(self):

        self.boundaries = list(
            self.original_boundaries
        )

        self.selected = None

        self.update_graphics()

        print(
            "RESET ALL COLLIMATORS"
        )



def correct_optical(bgr):
    """광카메라 프레임의 기울기와 중심 어긋남을 편다.

    회전과 평행이동을 **한 번의 warpAffine** 으로 처리한다. 두 번 나눠 걸면
    보간이 두 번 들어가 그림이 뭉개진다.

    가장자리는 검게 남는다. 조사야가 화면에 여백 60px 이상을 두고 들어오므로
    잘려 나가는 것은 조사야 밖이다.
    """
    if not OPTICAL_ROLL_DEG and not OPTICAL_DX_PX and not OPTICAL_DY_PX:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), OPTICAL_ROLL_DEG, 1.0)
    M[0, 2] += OPTICAL_DX_PX * (w / 640.0)
    M[1, 2] += OPTICAL_DY_PX * (h / 480.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class CameraManager:

    def __init__(self):

        self.running = True

        self.latest_thermal = None

        self.thermal_lock = (
            threading.Lock()
        )

        self.pipe = None

        self.thermal_camera = None

        self.landmarker = None

        self.worker_thread = None

        self.thermal_thread = None

        self.latest_display = None

        self.display_lock = (
            threading.Lock()
        )


    def start(self):

        print(
            "Starting cameras..."
        )

        # MediaPipe 는 CPU 를 코어 하나 넘게 먹어서 파이가 저전압에 걸린다.
        # 지금 단계에서는 그 결과가 어디에도 안 쓰이므로 끌 수 있게 해 둔다.
        # Hand PA 의 ROI 검출을 붙일 때 다시 켜면 된다.
        if os.environ.get("MLC_NO_MEDIAPIPE") == "1":
            self.landmarker = None
            print("MediaPipe OFF (MLC_NO_MEDIAPIPE=1) - CPU 부하를 줄입니다")
        else:
            self.landmarker = (
                HandLandmarker.create_from_options(
                    options
                )
            )


        cmd = [
            "rpicam-vid",
            "--nopreview",
            "--mode",
            OPTICAL_SENSOR_MODE,
            "-t",
            "0",
            "--inline",
            "--width",
            str(CAMERA_WIDTH),
            "--height",
            str(CAMERA_HEIGHT),
            "--codec",
            "mjpeg",
            "--framerate",
            "30",
            "-o",
            "-"
        ]

        self.pipe = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0
        )


        # 열화상은 코어가 혼자 붙들고, 그림만 소켓으로 받는다.
        # 장치 하나를 두 프로세스가 동시에 열 수 없다.
        self.thermal_camera = None
        self.mlc = MLCClient(MLC_HOST, MLC_PORT)
        self.mlc.start()
        print("MLC core client started (%s:%d)" % (MLC_HOST, MLC_PORT))


        self.thermal_thread = (
            threading.Thread(
                target=self.thermal_reader,
                daemon=True
            )
        )

        self.thermal_thread.start()


        self.worker_thread = (
            threading.Thread(
                target=self.camera_worker,
                daemon=True
            )
        )

        self.worker_thread.start()

        return True


    def thermal_reader(self):
        """코어가 보내 주는 열화상 그림을 받아 온다. 장치는 열지 않는다.

        코어는 초당 5장만 보냅니다. 그 사이에는 같은 그림이 돌아옵니다.
        """
        while self.running:
            frame = self.mlc.thermal_bgr()
            if frame is not None:
                with self.thermal_lock:
                    self.latest_thermal = frame
            time.sleep(0.05)


    def camera_worker(self):

        buffer = b""

        try:

            while self.running:

                if self.pipe is None:
                    break

                chunk = (
                    self.pipe.stdout.read(
                        16384
                    )
                )

                if not chunk:
                    break

                buffer += chunk

                while self.running:

                    a = buffer.find(
                        b"\xff\xd8"
                    )

                    if a == -1:

                        buffer = buffer[-1:]

                        break

                    b = buffer.find(
                        b"\xff\xd9",
                        a + 2
                    )

                    if b == -1:
                        break

                    jpg = buffer[
                        a:b + 2
                    ]

                    buffer = buffer[
                        b + 2:
                    ]

                    bgr = cv2.imdecode(
                        np.frombuffer(
                            jpg,
                            dtype=np.uint8
                        ),
                        cv2.IMREAD_COLOR
                    )

                    if bgr is None:
                        continue

                    # 카메라 기울기와 중심 어긋남을 여기서 한 번에 편다.
                    # 디코딩 직후에 펴야 아래로 내려가는 모든 것(MediaPipe,
                    # 외곽선, 코어로 보내는 마스크)이 같은 그림을 본다.
                    bgr = correct_optical(bgr)


                    with self.thermal_lock:

                        if (
                            self.latest_thermal
                            is None
                        ):
                            continue

                        thermal = (
                            self.latest_thermal.copy()
                        )


                    rgb = cv2.cvtColor(
                        bgr,
                        cv2.COLOR_BGR2RGB
                    )

                    if self.landmarker is None:
                        # 끈 상태. mp.Image 를 만드는 것 자체도 비용이라 같이 건너뛴다.
                        result = _NO_HANDS
                    else:
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                        result = self.landmarker.detect(mp_image)


                    thermal_display = cv2.resize(
                        thermal,
                        (
                            CAMERA_WIDTH,
                            CAMERA_HEIGHT
                        ),
                        interpolation=cv2.INTER_CUBIC
                    )


                    if result.hand_landmarks:

                        hand_landmarks = (
                            result.hand_landmarks[0]
                        )

                        points = []

                        for landmark in (
                            hand_landmarks
                        ):

                            x = int(
                                landmark.x *
                                CAMERA_WIDTH
                            )

                            y = int(
                                landmark.y *
                                CAMERA_HEIGHT
                            )

                            x = max(
                                0,
                                min(
                                    CAMERA_WIDTH - 1,
                                    x
                                )
                            )

                            y = max(
                                0,
                                min(
                                    CAMERA_HEIGHT - 1,
                                    y
                                )
                            )

                            points.append(
                                (x, y)
                            )

                        xs = [
                            p[0]
                            for p in points
                        ]

                        ys = [
                            p[1]
                            for p in points
                        ]

                        x_min = min(xs)
                        x_max = max(xs)

                        y_min = min(ys)
                        y_max = max(ys)

                        hand_width = (
                            x_max - x_min
                        )

                        hand_height = (
                            y_max - y_min
                        )

                        margin_x = int(
                            hand_width * 0.15
                        )

                        margin_top = int(
                            hand_height * 0.10
                        )

                        margin_bottom = int(
                            hand_height * 0.40
                        )

                        roi_x1 = max(
                            0,
                            x_min - margin_x
                        )

                        roi_x2 = min(
                            CAMERA_WIDTH - 1,
                            x_max + margin_x
                        )

                        roi_y1 = max(
                            0,
                            y_min - margin_top
                        )

                        roi_y2 = min(
                            CAMERA_HEIGHT - 1,
                            y_max + margin_bottom
                        )


                        thermal_h, thermal_w = (
                            thermal.shape[:2]
                        )

                        thermal_x1 = int(
                            roi_x1 *
                            thermal_w /
                            CAMERA_WIDTH
                        )

                        thermal_x2 = int(
                            roi_x2 *
                            thermal_w /
                            CAMERA_WIDTH
                        )

                        thermal_y1 = int(
                            roi_y1 *
                            thermal_h /
                            CAMERA_HEIGHT
                        )

                        thermal_y2 = int(
                            roi_y2 *
                            thermal_h /
                            CAMERA_HEIGHT
                        )

                        thermal_x1 = max(
                            0,
                            min(
                                thermal_w - 1,
                                thermal_x1
                            )
                        )

                        thermal_x2 = max(
                            thermal_x1 + 1,
                            min(
                                thermal_w,
                                thermal_x2
                            )
                        )

                        thermal_y1 = max(
                            0,
                            min(
                                thermal_h - 1,
                                thermal_y1
                            )
                        )

                        thermal_y2 = max(
                            thermal_y1 + 1,
                            min(
                                thermal_h,
                                thermal_y2
                            )
                        )

                        thermal_roi = (
                            thermal[
                                thermal_y1:thermal_y2,
                                thermal_x1:thermal_x2
                            ]
                        )

                        if thermal_roi.size != 0:

                            roi_h, roi_w = (
                                thermal_roi.shape[:2]
                            )

                            scale = min(
                                CAMERA_WIDTH /
                                roi_w,

                                CAMERA_HEIGHT /
                                roi_h
                            )

                            new_w = max(
                                1,
                                int(
                                    roi_w *
                                    scale
                                )
                            )

                            new_h = max(
                                1,
                                int(
                                    roi_h *
                                    scale
                                )
                            )

                            thermal_display = (
                                cv2.resize(
                                    thermal_roi,
                                    (
                                        new_w,
                                        new_h
                                    ),
                                    interpolation=(
                                        cv2.INTER_CUBIC
                                    )
                                )
                            )

                            canvas = np.zeros(
                                (
                                    CAMERA_HEIGHT,
                                    CAMERA_WIDTH,
                                    3
                                ),
                                dtype=np.uint8
                            )

                            x_offset = (
                                CAMERA_WIDTH -
                                new_w
                            ) // 2

                            y_offset = (
                                CAMERA_HEIGHT -
                                new_h
                            ) // 2

                            canvas[
                                y_offset:
                                y_offset + new_h,
                                x_offset:
                                x_offset + new_w
                            ] = (
                                thermal_display
                            )

                            thermal_display = (
                                canvas
                            )


                    rgb_thermal = cv2.cvtColor(
                        thermal_display,
                        cv2.COLOR_BGR2RGB
                    )

                    with self.display_lock:

                        self.latest_display = (
                            rgb_thermal.copy()
                        )

        except Exception as e:

            print(
                "Camera worker error:",
                e
            )

        finally:

            print(
                "Camera worker stopped."
            )


    def get_latest_frame(self):

        with self.display_lock:

            if self.latest_display is not None:
                return self.latest_display.copy()

        # 자체 카메라가 한 장도 못 받을 때는 코어가 보내 주는 그림을 쓴다.
        #
        # 광카메라는 한 프로그램만 열 수 있다. seg_feed 가 쥐고 있으면 여기서는
        # 영원히 프레임이 안 나와서 화면이 까맣게만 보인다 (2026-08-31 관측).
        # 코어가 보내는 그림에는 광카메라 영상 + 초록 체형 외곽선 + 흰 조사야
        # 사각형이 이미 그려져 있으므로, 방사선사가 봐야 할 것은 다 들어 있다.
        bgr = self.mlc.thermal_bgr()
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


    def stop(self):

        self.running = False

        if self.pipe is not None:

            try:

                if self.pipe.poll() is None:

                    self.pipe.terminate()

                    try:

                        self.pipe.wait(
                            timeout=2
                        )

                    except subprocess.TimeoutExpired:

                        self.pipe.kill()

                        self.pipe.wait()

            except Exception:
                pass

        if self.thermal_camera is not None:

            try:

                self.thermal_camera.release()

            except Exception:
                pass

        if self.worker_thread is not None:

            self.worker_thread.join(
                timeout=1
            )

        if self.thermal_thread is not None:

            self.thermal_thread.join(
                timeout=1
            )

        if self.landmarker is not None:

            try:

                self.landmarker.close()

            except Exception:
                pass

            self.landmarker = None

        print(
            "Camera stopped."
        )



class ThermalUI(App):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.camera_manager = None

        self.update_event = None


    def build(self):

        root = BoxLayout(
            orientation="horizontal",
            spacing=5,
            padding=5
        )


        thermal = BoxLayout(
            orientation="vertical",
            size_hint_x=0.62,
            padding=5
        )

        # 흰 바탕이라 글자색을 지정해야 한다. Label 기본값은 흰색이다.
        title = Label(
            text="THERMAL CAMERA",
            font_size=18,
            bold=True,
            color=IOS_LABEL,
            size_hint_y=None,
            height=35
        )

        thermal.add_widget(
            title
        )

        self.collimator_area = (
            CollimatorArea()
        )

        thermal.add_widget(
            self.collimator_area
        )

        root.add_widget(
            thermal
        )


        motor = BoxLayout(
            orientation="vertical",
            size_hint_x=0.22,
            padding=8,
            spacing=15
        )

        # UP/DOWN 은 가장 자주 누르고 되돌릴 수 있는 동작이라 중립색으로 둔다.
        # 여기에 색을 쓰면 정작 강조해야 할 STOP 이 묻힌다.
        up_button = RoundedButton(
            text="UP\n[size=40]\u25b2[/size]",
            font_size=36,
            fill=IOS_FILL,
            color=IOS_LABEL,
        )

        down_button = RoundedButton(
            text="DOWN\n[size=40]\u25bc[/size]",
            font_size=36,
            fill=IOS_FILL,
            color=IOS_LABEL,
        )

        # 누르는 순간 한 걸음, 계속 누르고 있으면 연속 이동, 떼면 멈춤.
        up_button.bind(
            on_press=self._press_up,
            on_release=self._release_button,
        )

        down_button.bind(
            on_press=self._press_down,
            on_release=self._release_button,
        )

        # AUTO / MANUAL 토글. UP 과 DOWN 사이에 둔다.
        # 글자가 지금 모드를 그대로 보여 주고, 누르면 반대로 바뀐다.
        # 반지름을 크게 줘 알약 모양으로 둔다. 모양만으로 성격이 다름을 알 수
        # 있게 하려는 것이다 — 이건 이동이 아니라 모드 전환이다.
        self.mode_button = RoundedButton(
            text="AUTO",
            font_size=30,
            size_hint_y=0.55,
            radius=22,
            fill=IOS_GREEN,
        )
        self.mode_button.bind(on_release=self.toggle_mode)

        motor.add_widget(
            up_button
        )

        motor.add_widget(
            self.mode_button
        )

        motor.add_widget(
            down_button
        )

        root.add_widget(
            motor
        )


        control = BoxLayout(
            orientation="vertical",
            size_hint_x=0.16,
            padding=8,
            spacing=15
        )


        # RESET 은 '알고 있는 기준 자세로 되돌린다' 는 뜻이라 파랑이다.
        # 빨강은 STOP 하나만 쓴다 — 화면에 빨강이 둘이면 급할 때 눈이 헤맨다.
        reset_button = RoundedButton(
            # 환자가 들어올 때 조사야를 활짝 여는 버튼. 이름이 동작이 아니라
            # **언제 쓰는지**를 말하도록 RESET 에서 바꿨다.
            text="POSITION\n[size=16](자리잡기)[/size]",
            font_size=20,
            # line_height 는 글자 크기에 비례한다. 이 버튼은 첫 줄이 20px 라
            # 클래스 기본값(0.82)을 쓰면 MANUAL(30px)보다 간격이 2/3 로 좁아진다.
            # 같은 **픽셀** 간격을 내려면 그 비율만큼 키워야 한다: 0.82 x 30/20.
            line_height=1.2,
            fill=IOS_BLUE,
        )


        stop_button = RoundedButton(
            # ⚠(U+26A0)는 나눔 폰트에 없어 □ 로 나온다. ✖(U+2716)를 쓴다.
            # ⚠️ ✖ ‼ ✕ 는 이 폰트에 **없어서 네모로 나온다.** 아이콘을 바꿀
            # 때는 U+E000 을 그린 비트맵과 비교해 실제 유무를 확인할 것.
            # 있는 것: ▲ ▼ ■ ● × ! ◆ ▣
            text="STOP\n[size=34]\u25a0[/size]",
            font_size=34,
            # ■ 는 글자와 달리 어센더 여백이 없어 바짝 붙어 보인다.
            # 계산상으로는 기본값이 맞지만 눈에는 좁아서 키운다.
            line_height=1.0,
            fill=IOS_RED,
        )

        reset_button.bind(
            on_release=self.reset
        )

        stop_button.bind(
            on_release=self.stop
        )

        control.add_widget(
            reset_button
        )

        control.add_widget(
            stop_button
        )

        root.add_widget(
            control
        )


        self.camera_manager = (
            CameraManager()
        )

        success = (
            self.camera_manager.start()
        )

        if not success:

            print(
                "Camera initialization failed."
            )


        self.update_event = (
            Clock.schedule_interval(
                self.update_thermal_display,
                1 / 30
            )
        )

        return root


    _last_toggle_t = 0.0
    TOGGLE_DEBOUNCE_S = 0.25

    def toggle_mode(self, button):
        """AUTO <-> MANUAL 전환.

        MANUAL 로 갈 때는 값을 주지 않습니다. 코어가 '지금 값 그대로 멈춤'으로
        받아들이므로 리프가 움직이지 않습니다. 자동으로 얼추 맞춰진 자리에서
        이어서 미세조정할 수 있습니다.

        ⚠️ 2026-09-07: **터치 한 번에 `on_release` 가 두 번 올라온다.**
        그래서 AUTO 로 갔다가 곧바로 MANUAL 로 되돌아와, 눌러도 아무 일이
        없는 것처럼 보였다. 로그에 증거가 남는다:

            AUTO — 카메라가 리프를 움직입니다
            MANUAL — UP/DOWN 으로 조절하세요     <- 같은 한 번의 터치

        이건 8/30 에 고친 `probesysfs` 중복과 **다른 것**이다. 그때는 좌표가
        어긋난 중복(UP 과 DOWN 동시)이었고, 이번은 **같은 좌표가 두 번**이다.
        `config.ini` 를 다시 건드리면 VNC 쓰는 팀원의 마우스가 위험하므로
        여기서 시간으로 막는다. 사람은 250ms 안에 모드를 두 번 바꾸지 않는다.
        """
        now = time.time()
        dt = now - self._last_toggle_t
        # 진단용: 두 번째 이벤트가 **얼마나 뒤에** 오는지 눈으로 봐야 한다.
        # 250ms 로는 안 걸렸으므로 간격이 그보다 길다.
        print("[toggle] +%.3fs" % dt)
        if dt < self.TOGGLE_DEBOUNCE_S:
            print("[toggle] 중복으로 보고 무시했습니다")
            return
        self._last_toggle_t = now

        mlc = getattr(self.camera_manager, "mlc", None)
        if mlc is None:
            print("코어에 안 붙었습니다")
            return
        st = mlc.state
        if st is None:
            print("코어 상태를 아직 못 받았습니다")
            return

        if st.get("manual"):
            mlc.auto()
            print("AUTO — 카메라가 리프를 움직입니다")
        else:
            mlc.send({"cmd": "manual"})      # 값 없이 = 지금 자리에서 멈춤
            print("MANUAL — UP/DOWN 으로 조절하세요")

    def _update_mode_button(self, st):
        """버튼 글자와 색을 지금 모드에 맞춘다."""
        btn = getattr(self, "mode_button", None)
        if btn is None:
            return
        if st is None:
            # 두 번째 줄은 markup 으로 작게. 지금 무슨 모드인지 한글로
            # 한 번 더 말해 주면 방사선사가 화면만 보고 판단할 수 있다.
            text = "NO LINK\n[size=17](연결 없음)[/size]"
            color = IOS_DIM
        elif st.get("manual"):
            text = "MANUAL\n[size=17](수동 작동 중)[/size]"
            color = IOS_ORANGE
        else:
            text = "AUTO\n[size=17](자동 작동 중)[/size]"
            color = IOS_GREEN
        if btn.text != text:
            btn.text = text
            # ⚠️ background_color 가 아니라 fill 이다. RoundedButton 은
            # background_color 를 투명 고정으로 쓰기 때문이다.
            btn.fill = color
            # 이 세 바탕(주황·초록·연회색)은 모두 밝아서 흰 글씨는 대비가
            # 2.1~2.3:1 밖에 안 된다(읽기 기준 4.5:1). 검정이면 8~9:1 이다.
            # 어두운 바탕인 POSITION(파랑)·STOP(빨강)은 흰 글씨 그대로 둔다.
            btn.color = IOS_LABEL

    def _sync_leaves_from_core(self):
        """코어가 보고한 개방거리를 화면 그림에 반영한다.

        자동 모드에서는 카메라가 리프를 움직이므로 버튼을 안 눌러도 그림이
        따라가야 합니다. 화면이 실제와 다르면 사람이 상태를 잘못 읽습니다.

        아래 행(5~8)은 코어가 모르므로 건드리지 않습니다. 두 번째 아두이노를
        붙이면 여기서 8칸 모두 반영하도록 늘리면 됩니다.
        """
        mlc = getattr(self.camera_manager, "mlc", None)
        st = None if mlc is None else mlc.state
        self._update_mode_button(st)
        if st is None:
            return

        opens = st.get("open_mm") or []
        pos = st.get("pos_mm") or []
        limit = float(st.get("travel_mm", 57.5)) or 57.5
        area = self.collimator_area

        changed = False
        for i in range(min(len(opens), len(area.boundaries))):
            # i 는 **코어 채널** 인덱스, cell 은 **화면 칸**이다. 둘이 다르다.
            cell = cell_of_ch(i)
            # **실제 위치**(pos_mm)를 우선 쓴다. open_mm 은 목표라서, 길게 눌러
            # 끝까지 가는 중에는 이미 끝값이라 그림만 먼저 튀어 버린다.
            # 액츄에이터 위치 -> 조사야 개방거리는 뒤집어야 한다 (나올수록 차폐).
            p = pos[i] if i < len(pos) else None
            v = (limit - float(p)) if p is not None else float(opens[i])
            frac = min(max(float(v) / limit, 0.0), 1.0) * 0.5
            # 화면 윗줄은 위쪽 가장자리에서 내려오고, 아랫줄은 아래쪽에서
            # 올라온다. 둘을 같게 그리면 아래 리프가 중앙선을 넘어간다.
            want = (0.5 + frac) if cell < 4 else (0.5 - frac)
            if abs(area.boundaries[cell] - want) > 1e-4:
                area.boundaries[cell] = want
                changed = True
        if changed:
            area.update_graphics()


    def update_thermal_display(
        self,
        dt
    ):

        # 카메라 프레임이 없어도 리프 상태는 갱신해야 하므로 맨 앞에서 부른다.
        self._sync_leaves_from_core()

        if self.camera_manager is None:
            return

        frame = (
            self.camera_manager.get_latest_frame()
        )

        if frame is None:
            return

        height, width = (
            frame.shape[:2]
        )

        texture = Texture.create(
            size=(
                width,
                height
            ),
            colorfmt="rgb"
        )

        texture.blit_buffer(
            frame.tobytes(),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        texture.flip_vertical()

        self.collimator_area.set_thermal_frame(
            texture
        )


    STEP_MM = 1.0          # 버튼 한 번에 움직일 조사야 개방거리 [mm]
                           # 펌웨어는 0.3mm 부터 움직이므로 더 줄여도 된다.

    HOLD_DELAY_S = 0.35     # 이만큼 누르고 있으면 연속 이동 시작
    HOLD_PERIOD_S = 0.08    # 연속 이동 중 한 걸음 간격 (1mm x 12.5Hz = 12.5mm/s)

    def _press_up(self, button):
        self._press("up")

    def _press_down(self, button):
        self._press("down")

    def _press(self, direction):
        """누르는 순간 한 걸음 가고, 계속 누르고 있으면 연속 이동을 예약한다."""
        self._cancel_hold()
        self._step(direction)
        self._hold_ev = Clock.schedule_once(
            lambda dt: self._start_hold(direction), self.HOLD_DELAY_S)

    def _start_hold(self, direction):
        """누르고 있는 동안은 **목표를 끝까지 한 번만** 준다.

        1mm 씩 잘게 보내면 모터가 매번 '출발-구동-정지'를 반복해 뚝뚝 끊긴다.
        끝까지 목표를 주면 모터가 쉬지 않고 달리므로 부드럽다.
        손을 떼면 hold_stop 이 지금 실제 위치를 목표로 잡아 그 자리에 세운다.
        """
        sel = self.collimator_area.selected
        if sel is None:
            return
        mlc = getattr(self.camera_manager, "mlc", None)
        st = None if mlc is None else mlc.state
        if st is None:
            return

        limit = float(st.get("travel_mm", 57.5))
        up = (direction == "up")
        opening = up if sel < 4 else (not up)

        if sel >= int(st.get("n_ch", 4) or 4):
            # 코어가 모르는 채널이다(--no-bottom). 화면만 끝까지 보낸다.
            if not hasattr(self, "_sim_open"):
                self._sim_open = {}
            self._sim_open[sel] = limit if opening else 0.0
            frac = (0.5 if opening else 0.0)
            self.collimator_area.boundaries[sel] = 0.5 - frac
            self.collimator_area.update_graphics()
            return

        mlc.manual(motor_no(sel), limit if opening else 0.0)
        self._holding = True
        print("%d번 리프 연속 %s" % (motor_no(sel), "열림" if opening else "닫힘"))

    def _release_button(self, button):
        self._cancel_hold()
        if getattr(self, "_holding", False):
            self._holding = False
            mlc = getattr(self.camera_manager, "mlc", None)
            if mlc is not None:
                mlc.send({"cmd": "hold_stop"})   # 지금 있는 자리에서 멈춘다
                print("멈춤")

    def _cancel_hold(self):
        ev = getattr(self, "_hold_ev", None)
        if ev is not None:
            ev.cancel()
            self._hold_ev = None

    def move_up(self, button):
        self._step("up")


    def move_down(self, button):
        self._step("down")

    def _step(self, direction):
        """한계를 넘지 않을 때만 화면과 실제 리프를 **함께** 움직인다.

        리프는 행정 한계(코어가 알려주는 travel_mm, 현재 57.5mm)를 넘어갈 수 없습니다.
        그 값은 조사야 높이 115mm 의 절반이라, 위·아래 리프가 각각 그만큼 나오면
        정확히 맞닿습니다. 더 나가면 서로 부딪칩니다.

        한계에 닿으면 **화면 그림도 움직이지 않습니다.** 그림만 더 가면 실제보다
        더 열렸다고 착각하게 되기 때문입니다.

        5~8번(아래 뱅크)도 두 번째 아두이노로 실제로 움직입니다. 코어를
        --no-bottom 으로 띄웠을 때만 표시 전용으로 떨어집니다.
        """
        sel = self.collimator_area.selected
        if sel is None:
            print("리프를 먼저 고르세요")
            return

        # 화면에서의 위/아래를 '개방/차폐'로 옮긴다.
        # 위 행은 위쪽 가장자리에서 내려와 가리므로 **위로 갈수록 열리고**,
        # 아래 행은 아래쪽 가장자리에서 올라와 가리므로 **아래로 갈수록 열린다**.
        # (Case_Assembly_10 실측: F행 y 53~113, R행 y 112~172 에서 만난다)
        # 이걸 같게 두면 아래 리프가 중앙선을 넘어 올라간다.
        up = (direction == "up")
        opening = up if sel < 4 else (not up)
        delta_mm = self.STEP_MM if opening else -self.STEP_MM

        mlc = getattr(self.camera_manager, "mlc", None)
        st = None if mlc is None else mlc.state
        limit = 57.5 if st is None else float(st.get("travel_mm", 57.5))

        # 코어가 보고한 채널 수까지가 실제로 움직인다. 2026-09-06 에 보드가
        # 두 대가 되어 8이 되었다 — 여기에 4 를 박아두면 그때부터 거짓말이 된다.
        driven = sel < int((st or {}).get("n_ch", 4) or 4)
        if driven:
            if st is None:
                print("코어에 아직 안 붙었습니다 — 움직이지 않습니다")
                return
            opens = st.get("open_mm") or []
            ci = ch_index(sel)
            if ci >= len(opens):
                print("코어가 보고한 채널 수가 모자랍니다")
                return
            cur = float(opens[ci])
        else:
            # 코어가 모르는 채널이다(--no-bottom 으로 띄웠을 때).
            # 그때만 GUI 가 스스로 위치를 기억해 그림을 움직인다.
            if not hasattr(self, "_sim_open"):
                self._sim_open = {}
            cur = float(self._sim_open.get(sel, 0.0))

        want = min(max(cur + delta_mm, 0.0), limit)
        if abs(want - cur) < 0.01:
            edge = "완전 개방" if delta_mm > 0 else "완전 차폐"
            print("%d번 리프는 이미 한계입니다 (%s, %.1f/%.1fmm)"
                  % (motor_no(sel), edge, cur, limit))
            return

        if driven:
            mlc.manual(motor_no(sel), want)
            tail = ""
        else:
            self._sim_open[sel] = want
            tail = "   (표시 전용 — 이 리프에는 아직 모터가 없습니다)"

        # 그림을 mm 에서 직접 계산한다. 버튼마다 조금씩 더하는 방식은
        # mm 와 그림이 어긋나 리프가 중앙선을 넘어 그려지게 만든다.
        # (모델링의 누설 대비 겹침은 화면에 반영하지 않는다 — 0mm = 중앙선)
        frac = min(max(want / limit, 0.0), 1.0) * 0.5
        self.collimator_area.boundaries[sel] = (0.5 + frac) if sel < 4 else (0.5 - frac)
        self.collimator_area.update_graphics()

        print("%d번 리프 %s %.1f -> %.1f mm (한계 %.1f)%s"
              % (motor_no(sel), "열림" if delta_mm > 0 else "닫힘",
                 cur, want, limit, tail))
        return True          # 연속 이동을 이어가도 좋다는 뜻


    def reset(self, button):
        """모든 축을 끝까지 물려 조사야를 최대로 연다.

        액츄에이터 로드가 전부 들어간 상태(= 기계적 원점)이고, 리프가 조사야
        밖으로 완전히 물러난 자리다. 환자를 세우기 전 '아무것도 가리지 않은'
        기준 자세로 쓴다.

        ⚠️ 2026-09-07 에 **완전 차폐에서 최대 개방으로 뜻이 뒤집혔다.**
        이 버튼은 이제 조사야가 가장 넓어지는 동작이다.

        ⚠️ 2026-09-07 밤 (1차): 동작을 원점 복귀(HOME)로 바꿨다가,
        ⚠️ 2026-09-07 밤 (2차): **다시 단순 이동으로 되돌렸다. 사용자 지정이다.**
        HOME 은 뱅크를 **하나씩 차례로** 돈다(앞행 12초 → 뒷행 12초 = 약 24초).
        앞행이 움직이다 멈추고 뒷행이 움직이는 게 눈에 보여서 길게 느껴진다.
        `manual_all` 은 8채널 목표를 **한 번에** 내보내므로 두 행이 같이 움직인다.

        대가: 원점을 다시 잡지 않으므로 **개루프 누적 오차가 안 지워진다.**
        오차가 의심되면 소켓으로 `{"cmd":"home"}` 을 따로 걸 것.

        채널 수와 행정은 코어가 보고한 값을 쓴다. 8을 박아 두면 코어를
        --no-bottom 으로 띄웠을 때 배열 길이가 안 맞아 거부당한다.
        """
        self.collimator_area.reset_collimators()
        mlc = getattr(self.camera_manager, "mlc", None)
        if mlc is not None:
            st = mlc.state or {}
            n = int(st.get("n_ch", 8) or 8)
            limit = float(st.get("travel_mm", 57.5) or 57.5)
            # manual_all 은 수동 전환까지 겸한다. 카메라가 곧바로 다시 덮어쓰면
            # '열렸다' 를 눈으로 확인할 시간이 없기 때문이다.
            #
            # 개방거리 = limit(travel_mm) 가 **로드 완전 수축 = 조사야 최대 개방**
            # 이다 — [[mlc-leaf-polarity]]. 8채널이 한 배열로 나가므로 앞행·뒷행이
            # 동시에 움직인다.
            mlc.manual_all([limit] * n)
            print("POSITION — 리프 %d장 전부 수축, 조사야 완전 개방 (%.1fmm)"
                  % (n, limit))
        else:
            print("RESET — 코어에 안 붙어 있습니다 (화면만 되돌립니다)")
        # 대체 기억도 지운다. 안 지우면 화면은 되돌아갔는데 숫자만 남아
        # 다음 UP 이 엉뚱한 데서 출발한다.
        self._sim_open = {}


    def stop(self, button):
        """모든 축의 움직임을 그 자리에서 멈춘다. **앱은 계속 돈다.**

        코어의 `stop` 은 진행 중인 목표를 버리고 모터 출력을 끊는다. 리프는
        지금 있는 자리에 그대로 선다.

        ⚠️ 예전에는 여기서 `camera_manager.stop()` 까지 불러 **카메라가 죽고
        앱이 같이 내려갔다.** 급할 때 누르는 버튼이 화면을 꺼 버리면 무슨 일이
        벌어졌는지 볼 수가 없다. 2026-09-07 에 떼어냈다. 앱을 끝내는 것은 창을
        닫는 일이고, 그때는 on_stop() 이 카메라를 정리한다.

        ⚠️ 코어의 stop 은 모터 출력까지 끊으므로(motor=False), 다시 움직이려면
        소켓으로 {"cmd":"motor","on":true} 를 보내야 한다. GUI 에는 아직 모터를
        켜는 버튼이 없다.
        """
        # 꾹 누르고 있던 중이었다면 그 연속 이동부터 끊는다. 안 그러면 손을
        # 뗄 때 hold_stop 이 뒤늦게 날아가 멈춘 축을 다시 건드린다.
        self._cancel_hold()
        self._holding = False

        mlc = getattr(self.camera_manager, "mlc", None)
        if mlc is not None:
            mlc.stop_motor()
            print("STOP — 모든 축 정지 (모터 출력 off)")
        else:
            print("STOP — 코어에 안 붙어 있습니다")


    def on_stop(self):

        if self.update_event is not None:

            self.update_event.cancel()

        if self.camera_manager is not None:

            self.camera_manager.stop()



if __name__ == "__main__":

    ThermalUI().run()
