# -*- coding: utf-8 -*-
"""
leaves.py — 체온 외곽선을 리프 4개의 개방 거리로 바꾸는 계산

좌표 약속 (field.py 의 조사야 격자 위에서)
    격자 = 230 x 190 px = 115 x 95 mm,  0.5 mm/px
    가로를 4등분한 것이 리프 4장 (CH1=맨 왼쪽 ... CH4=맨 오른쪽)
    가로 중심선(y = h/2)이 상하 뱅크가 맞물리는 자리

'개방 거리(open_mm)' 정의 — 상단 뱅크 기준
    중심선에서 리프 끝단까지의 거리 [mm]
      0 mm      = 리프가 중심선까지 완전히 닫힘 (그 열 전체 차폐)
      57.5 mm   = 리프가 끝까지 물러남 (그 열 전체 개방)

액츄에이터 변환
    홈(0 mm, 로드 완전 수축) = 리프 완전히 닫힘 이므로
        액츄에이터 위치 [mm] = 개방 거리 [mm]
    통신이 끊기면 아두이노 워치독이 0 mm 로 보내므로 자동으로 완전 차폐가 됩니다.
    이 방향이 안전한 쪽입니다.

융합 판정
    '무엇이 사람인가'는 열화상이 정합니다 — 그게 물리적으로 타당한 신호입니다.
    '그 경계가 정확히 어디인가'만 광학 엣지로 다듬습니다. Lepton 은 160x120 이라
    격자에서 1픽셀이 1.2mm 쯤 되는데, 광학은 그 자리를 0.15mm 로 봅니다.
    그래서 열화상 경계에서 +-몇 mm 안에 있는 가장 강한 광학 엣지로 스냅합니다.
"""

import numpy as np
import cv2

# ======================================================================
#  설정
# ======================================================================

N_COLS = 4                  # 리프 개수 (= 아두이노 채널 수)

# 리프 행정 [mm]. 0 = 로드 완전 수축(조사야 개방), 이 값 = 완전 전진(완전 차폐).
#
# 2026-08-31 Case_Assembly_10 실측으로 60.0 으로 고쳤다. 그전 57.5 는
# "조사야 세로 115mm 의 절반" 이라는 추정이었는데, 설계는 이렇다:
#
#   Blade_F1~F4 : y  53.2 ~ 113.2   (길이 60mm)   <- 위 행
#   Blade_R1~R4 : y 111.8 ~ 171.8   (길이 60mm)   <- 아래 행
#                 두 행이 y≈112.5 에서 1.4mm 겹쳐 만난다 (누설 대비 설계)
#
# 액츄에이터 가용 행정도 60mm 라 딱 맞는다. 이 값은 펌웨어 LIM, 실패안전 위치,
# 종료 시 차폐 위치, GUI 의 한계까지 전부를 결정한다.
LEAF_TRAVEL_MM = 60.0

# 인체 보호 마진 [mm] — 검출된 체온 경계보다 이만큼 더 열어둠
MARGIN_MM = 5.0

# 홈(0mm)에서 리프가 완전히 닫히는가? (2026-08-27 사용자 확인: 그렇다)
# 반대로 조립했다면 True 로 바꾸세요. 그러면 액츄에이터 = 행정 - 개방거리.
# 로드가 나올수록 리프가 조사야 안으로 들어와 가린다 (2026-08-30 실물로 확인:
# CH1 을 30mm 내보내니 조사야 구멍이 좁아졌다). 따라서 액츄에이터 위치는
# '개방거리'가 아니라 '차폐량'이므로 뒤집어서 보내야 한다.
#   액츄에이터 0mm(로드 완전 수축) = 조사야 완전 개방
#   액츄에이터 57.5mm(로드 최대 전진) = 조사야 완전 차폐
INVERT_LEAF = True

# --- 온도 판정 --------------------------------------------------------
# 옷을 입은 사람의 표면 온도는 생각보다 낮습니다(28~33도). 노출 피부는 31~36도.
TEMP_MIN_C = 28.0        # 2026-08-28 실측 튜닝값. 옷 입은 사람 표면은 30도에 못 미친다
TEMP_MAX_C = 41.0

MIN_RUN_PX = 3        # 세로로 이만큼 이어져야 '인체 시작'으로 인정 (점 노이즈 제거)

# 덩어리 하나가 이 넓이는 되어야 인체로 인정 [mm^2].
# ★ '가장 큰 덩어리만' 쓰면 안 됩니다. 손가락, 팔, 어깨는 열화상에서 몸통과
#   끊어진 별개 덩어리로 잡히는 일이 흔한데, 그걸 버리면 그 위로 리프가
#   그대로 내려와 사람을 자릅니다. 2026-08-27 실기에서 손가락이 잘렸습니다.
#   그래서 '제일 큰 것만'이 아니라 '너무 작은 것만' 버립니다.
MIN_BLOB_MM2 = 15.0

# 한 열에 이만큼은 있어야 그 열에 인체가 있다고 인정 [mm^2].
# 낮게 잡습니다. 여기서 인색하게 굴면 리프가 사람을 침범합니다.
MIN_BAND_MM2 = 4.0

# --- 광학 융합 --------------------------------------------------------
SNAP_RADIUS_MM = 6.0        # 열화상 경계에서 이 범위 안의 광학 엣지만 채택
SNAP_EDGE_RATIO = 0.35      # 프레임 최대 엣지의 이 비율은 넘어야 '진짜 경계'
SNAP_PROMINENCE = 0.25      # 창 안에서 이만큼은 튀어야 '봉우리'로 인정
SNAP_SMOOTH_PX = 5          # 엣지를 가로로 이만큼 뭉개서 한 픽셀 노이즈에 안 흔들리게
PROFILE_MEDIAN_PX = 9       # 최종 외곽선 프로파일 중앙값 필터 폭 (약 4.5mm)

# --- 시간 평활 --------------------------------------------------------
EMA_ALPHA = 0.40            # 0에 가까울수록 부드럽지만 느림, 1이면 평활화 없음
SEND_DEADBAND_MM = 0.8      # 이보다 작은 변화는 보내지 않음
MAX_RATE_MM_S = 15.0        # 목표값이 튀는 속도 상한. 액츄에이터 실속도(약 13mm/s)와 비슷하게.

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


# ======================================================================
#  1. 체온 마스크
# ======================================================================

# 검사 부위에서 이만큼 넘게 떨어진 덩어리는 노이즈로 본다 [mm].
# 0 이면 이 걸러내기를 끈다.
MAIN_JOIN_MM = 25.0


def _keep_main_blob(m, mm_per_px, join_mm):
    """격자 중앙의 덩어리와 그 근처만 남긴다. 먼 것은 노이즈로 버린다.

    ⚠️ '제일 큰 것만 남기기'(largest_only)와 다르다. 그 방식은 팔이 잡음으로
    끊기면 팔을 통째로 버리고 **그 자리에 리프를 내려보낸다** — 인체를 덮는
    방향이라 위험하다. 여기서는 가까이 있는 조각을 살린다.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:                                  # 배경 + 덩어리 하나면 그대로
        return m
    h, w = m.shape[:2]
    main = int(labels[h // 2, w // 2])
    if main == 0:                               # 중앙이 비었으면 가장 큰 것
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    r = max(1, int(round(join_mm / mm_per_px)))
    grown = cv2.dilate((labels == main).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        if i == main or np.any(grown[labels == i]):
            keep[i] = 255
    return keep[labels]


def body_mask(temp_c, tmin=TEMP_MIN_C, tmax=TEMP_MAX_C, mm_per_px=0.5,
              min_blob_mm2=MIN_BLOB_MM2, largest_only=False,
              relative=False, bias=0, join_mm=None):
    """조사야 격자 위의 온도 배열 -> 인체 마스크 (0/255 uint8).

    largest_only 는 기본이 False 입니다. True 로 두면 몸통만 남기고 손가락·팔을
    버려서 그 자리에 리프가 내려옵니다. 배경에 뜨거운 물체(전등, 라디에이터)가
    있어 그것까지 인체로 잡힐 때만 잠깐 켜세요.
    """
    if relative:
        # AGC(8비트) 모드: 절대 온도가 없으므로 '화면에서 유난히 밝은 덩어리'를 인체로 본다.
        lo, hi = float(np.percentile(temp_c, 1)), float(np.percentile(temp_c, 99))
        g = np.clip((temp_c - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        thr, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = int(max(1, min(254, int(thr) + bias)))
        m = (g >= thr).astype(np.uint8) * 255
    else:
        m = ((temp_c >= tmin) & (temp_c <= tmax)).astype(np.uint8) * 255

    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K3)     # 점 노이즈 제거
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K9)    # 구멍 메우기 + 끊긴 부분 잇기

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if largest_only:
            keep = np.zeros(areas.size, bool)
            keep[int(np.argmax(areas))] = True
        else:
            # 너무 작은 것만 버린다. 손가락 하나도 남겨야 한다.
            min_px = max(1, int(round(min_blob_mm2 / (mm_per_px ** 2))))
            keep = areas >= min_px
            if not keep.any():           # 전부 작으면 그래도 제일 큰 건 남긴다
                keep[int(np.argmax(areas))] = True
        lut = np.zeros(n, np.uint8)
        lut[1:][keep] = 255
        m = lut[labels]

    # 검사 부위에서 멀리 떨어진 덩어리 버리기. 2026-09-07 AUTO 시험에서
    # 구석 조각 때문에 4·8번 열이 활짝 열린 채 남는 것을 보고 넣었다.
    j = MAIN_JOIN_MM if join_mm is None else join_mm
    if j and j > 0 and not largest_only:
        m = _keep_main_blob(m, mm_per_px, j)
    return m


# ======================================================================
#  2. 외곽선 프로파일 (열마다 '인체의 가장 윗점')
# ======================================================================

def top_profile(mask, min_run_px=MIN_RUN_PX):
    """각 가로 위치 x 에서 인체가 시작되는 행 y 를 구한다.

    반환: 길이 w 의 int 배열. 인체가 없는 열은 -1.

    한 픽셀짜리 잡음에 속지 않도록 '세로로 min_run_px 만큼 이어진 곳'이
    시작이라고 봅니다. 세로 방향 침식 한 번으로 그 조건을 만듭니다.
    """
    k = max(1, int(min_run_px))
    if k > 1:
        ker = np.ones((k, 1), np.uint8)
        solid = cv2.erode(mask, ker)
    else:
        solid = mask

    b = solid > 0
    has = b.any(axis=0)
    first = np.argmax(b, axis=0)            # 없으면 0 이 나오므로 has 로 걸러야 함
    # 침식으로 밀린 만큼(k//2) 되돌리되 **0 아래로는 내리지 않는다.**
    #
    # (2026-08-31 버그) 여기가 그냥 `first - k // 2` 였다. 인체가 화면 맨 위
    # (0행)부터 있으면 0 - 1 = -1 이 되는데, -1 은 '이 열에는 사람이 없다'는
    # 표식이라 다음 단계에서 통째로 걸러졌다. 그 결과 **환자가 조사야를 위까지
    # 꽉 채울수록 리프가 닫히는** 정반대 거동이 나왔다. 흉부처럼 몸이 조사야를
    # 가득 채우는 검사에서 조사야를 막아버리는 방향이라 그냥 둘 수 없다.
    y = np.where(has, np.maximum(first - k // 2, 0), -1)
    return np.clip(y, -1, mask.shape[0] - 1).astype(np.int32)


def edge_map(optical_grid):
    """광학 영상 -> 0~1 로 정규화한 엣지 세기."""
    if optical_grid is None:
        return None
    g = optical_grid
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    # 가로로 살짝 뭉개서 세로 경계선이 한 열에만 걸리는 일을 막는다.
    if SNAP_SMOOTH_PX > 1:
        mag = cv2.blur(mag, (SNAP_SMOOTH_PX, 1))
    hi = float(np.percentile(mag, 99.5))
    return mag / max(hi, 1e-6)


def snap_to_edges(profile, edges, mm_per_px, radius_mm=SNAP_RADIUS_MM,
                  min_ratio=SNAP_EDGE_RATIO, min_prom=SNAP_PROMINENCE):
    """열화상 경계를 근처의 광학 엣지로 당긴다.

    profile: top_profile() 결과, edges: edge_map() 결과.
    반환: (보정된 profile, 스냅된 열의 비율)

    주의 — 창 안에서 그냥 '가장 센 엣지'를 고르면 안 됩니다.
    인체 외곽이 거의 수직인 자리(어깨 옆구리 같은 곳)에서는 세로로 훑어도
    엣지가 창 전체에 고르게 깔려 봉우리가 없고, argmax 가 창 끝으로 밀려서
    경계가 6mm 씩 튑니다. 실제로 그 증상을 확인하고 아래 두 가지를 넣었습니다.

      1) 거리 가중 — 열화상이 짚은 자리에서 멀수록 점수를 깎습니다.
         '경계를 다듬는' 것이지 '경계를 다시 찾는' 것이 아니기 때문입니다.
      2) 돌출도 검사 — 창 안 중앙값보다 확실히 튀어야 봉우리로 인정합니다.
         평평하면 스냅하지 않고 열화상 값을 그대로 씁니다.
    """
    if edges is None:
        return profile, 0.0

    h = edges.shape[0]
    r = max(1, int(round(radius_mm / mm_per_px)))
    valid = np.where(profile >= 0)[0]
    if valid.size == 0:
        return profile, 0.0

    offs = np.arange(-r, r + 1)
    rows = profile[valid][:, None] + offs[None, :]          # (n, 2r+1)
    inside = (rows >= 0) & (rows < h)
    rows_c = np.clip(rows, 0, h - 1)
    cols = np.repeat(valid[:, None], offs.size, axis=1)

    vals = np.where(inside, edges[rows_c, cols], -1.0)

    sigma = max(1.0, r / 2.0)
    weight = np.exp(-0.5 * (offs / sigma) ** 2)
    best = np.argmax(vals * weight[None, :], axis=1)

    idx = np.arange(valid.size)
    peak = vals[idx, best]
    # offset 0 은 언제나 창 안이므로 전부 NaN 인 행은 생기지 않는다.
    base = np.nanmedian(np.where(inside, vals, np.nan), axis=1)
    took = (peak >= min_ratio) & (peak - base >= min_prom)

    out = profile.copy()
    out[valid[took]] = rows_c[idx, best][took]
    return out, float(took.mean())


def smooth_profile(profile, k=PROFILE_MEDIAN_PX):
    """유효한 값만 골라 중앙값 필터. 인체가 없는 열(-1)은 그대로 둔다."""
    if k < 3:
        return profile
    p = profile.astype(np.float32).copy()
    valid = p >= 0
    if valid.sum() < k:
        return profile
    # 인체가 없는 열은 중앙값 계산 전에 양옆 값으로 메웁니다. NaN 을 섞으면
    # 창 하나가 통째로 NaN 인 경우가 생겨 경고가 뜨고 값도 못 씁니다.
    idx = np.arange(p.size)
    p = np.interp(idx, idx[valid], p[valid])

    pad = k // 2
    win = np.lib.stride_tricks.sliding_window_view(np.pad(p, pad, mode="edge"), k)
    med = np.median(win, axis=1)
    return np.where(valid, np.rint(med), -1.0).astype(np.int32)


# ======================================================================
#  3. 열 -> 리프 개방 거리
# ======================================================================

def openings_from_profile(profile, mask, grid_h, grid_w, mm_per_px,
                          n_cols=N_COLS, margin_mm=MARGIN_MM,
                          leaf_max_mm=LEAF_TRAVEL_MM, min_band_mm2=MIN_BAND_MM2):
    """리프별 개방 거리 [mm] 와 진단 정보를 돌려준다.

    반환: (openings[n_cols], info[n_cols])
      info[c] = {"y_top": 화면 행 또는 None, "pix": 그 열의 인체 픽셀 수}
    """
    mid = grid_h // 2
    limit = min(float(leaf_max_mm), mid * mm_per_px)
    min_band_pix = max(1, int(round(min_band_mm2 / (mm_per_px ** 2))))

    openings = []
    infos = []
    for c in range(n_cols):
        x0 = (c * grid_w) // n_cols
        x1 = ((c + 1) * grid_w) // n_cols

        band = mask[:, x0:x1]
        pix = int(np.count_nonzero(band))
        seg = profile[x0:x1]
        seg = seg[seg >= 0]

        if pix < min_band_pix or seg.size == 0:
            # 그 열에 인체가 없으면 완전 차폐. 찍을 것이 없으니 닫는 게 맞습니다.
            openings.append(0.0)
            infos.append({"y_top": None, "pix": pix, "clamped": False})
            continue

        y_top = int(seg.min())                    # 그 열에서 가장 위에 있는 인체 점
        d_mm = (mid - y_top) * mm_per_px          # 중심선 위로 얼마나 올라와 있나
        want = d_mm + margin_mm
        open_mm = float(np.clip(want, 0.0, limit))
        # want > limit 면 인체가 조사야 위쪽 경계를 넘어간 것입니다. 리프를 끝까지
        # 물려도 그 부위는 못 덮습니다. 침범은 아니지만 사람이 알아야 합니다.
        openings.append(open_mm)
        infos.append({"y_top": y_top, "pix": pix, "clamped": want > limit + 1e-6})

    return openings, infos


# 날 길이 편차 보정 [mm] — 채널 1~8 순서 (0 = 편차 없음)
#
# 2026-09-07: 모델링 오류로 **4번과 8번 날이 5mm 길다** (실측·도면 일치).
# 날이 길면 같은 액츄에이터 위치에서 그만큼 더 가린다. 그래서 그 채널은
# 액츄에이터를 5mm 덜 내보내야 다른 채널과 개방거리가 같아진다.
#
# ⚠️ 완전 개방일 때는 보정이 다 안 들어간다. 액츄에이터가 0 아래로 못 가기
#    때문이다. 즉 **활짝 열었을 때만 4·8번이 5mm 덜 열린다** — 날이 긴 만큼
#    끝까지 물러나지 못하는 물리적 사실이라 소프트웨어로 못 고친다.
#
# 날을 다시 만들어 길이를 맞추면 전부 0.0 으로 되돌릴 것.
# ⚠️ 2026-09-07: 5.0 으로 넣었다가 **되돌렸다.**
#
# 4·8번 날이 5mm 길어서 "액츄에이터를 5mm 덜 보내면 된다"고 추론했는데,
# 실제로 55mm 에서 **중앙선에 못 닿아 차폐가 안 됐다.** 날의 여분 5mm 가 끝
# 쪽 도달거리로 그대로 이어지지 않는다는 뜻이다(마운트 쪽이 길거나 폭 방향).
#
# 차폐에서 **덜 가리는 쪽 실수는 절대 안 된다.** 보정 없이 두면 4·8번이 5mm
# 더 가리는데(과잉 차폐), 그건 안전한 방향이다. 그래서 0 으로 둔다.
#
# 다시 넣으려면 **완전 차폐가 되는지 눈으로 확인한 뒤에만** 할 것.
# 2026-09-07 오후: **8번만 행정이 5mm 길다** (사용자 실측). 같은 명령에 5mm
# 더 나가므로 액츄에이터 목표에서 그만큼 뺀다.
#
# ⚠️ 어제 4·8번에 넣었다가 되돌린 적이 있다. 그때는 "4·8번 차폐 실패" 보고를
# 받고 넣었는데 실제로는 **3·7번**이었다 — 잘못된 대상에 적용한 것이었다.
# 이번엔 8번만 넣는다.
#
# ⚠️ 넣은 뒤 **완전 차폐에서 빛이 새는지 반드시 눈으로 확인할 것.**
# 덜 가리는 실수는 차폐에서 절대 하면 안 된다. 새면 즉시 0 으로 되돌린다.
# ⚠️ 2026-09-07 밤: 8번에 남아 있던 5.0 을 0 으로 되돌렸다.
# 4·8번 날이 5mm 길다고 보고 넣은 보정인데, 4번은 차폐 부족 때문에
# 이미 0 으로 되돌렸으면서 **8번만 남아 있었다.** 그 탓에 완전 차폐를
# 명령해도 8번 축이 55mm 에서 멈춰 빛이 샜다.
# 차폐 부족은 안전하지 않은 방향이므로, 확신이 없으면 0 이 맞다.
BLADE_EXTRA_MM = [0.0] * 8


def to_actuator(open_mm, leaf_max_mm=LEAF_TRAVEL_MM, invert=INVERT_LEAF, ch=None):
    """개방거리[mm] -> 액츄에이터 목표[mm].

    `ch` 는 **1부터 시작하는 채널 번호**다. 주면 그 채널의 날 길이 편차를
    빼 준다. 안 주면 보정 없이 예전처럼 동작한다 (기존 호출부 호환).

    날의 기하가 들어가는 지점은 여기뿐이다. 펌웨어에 넣으면 mlc8.py 같은
    진단 도구가 보정 없이 돌아 헷갈리고, GUI 에 넣으면 화면이 실제와 달라진다.
    """
    v = (leaf_max_mm - open_mm) if invert else open_mm
    if ch is not None and 1 <= ch <= len(BLADE_EXTRA_MM):
        v -= BLADE_EXTRA_MM[ch - 1]
    return float(np.clip(v, 0.0, leaf_max_mm))


# ======================================================================
#  4. 한 프레임 통째로 푸는 진입점
# ======================================================================

class LeafSolver:
    """프레임 하나를 받아 채널 4개의 목표 위치 [mm] 를 내놓는다.

    시간 평활(EMA), 변화율 제한, 데드밴드까지 여기서 처리합니다.
    호출 쪽은 solve() 가 돌려준 targets 를 그대로 SET 으로 보내면 됩니다.
    """

    def __init__(self, grid_h, grid_w, mm_per_px, n_cols=N_COLS,
                 fusion=True, relative=False):
        self.h, self.w = grid_h, grid_w
        self.mm_px = mm_per_px
        self.n = n_cols
        self.fusion = fusion
        self.relative = relative

        self.tmin, self.tmax = TEMP_MIN_C, TEMP_MAX_C
        self.margin = MARGIN_MM
        self.largest_only = False        # 손가락·팔을 버리지 않는다. 위 주석 참고.
        self.bias = 0

        self.smoothed = [0.0] * n_cols      # EMA 를 거친 개방 거리
        self.sent = [None] * n_cols         # 마지막으로 실제 전송한 값
        self.last_t = None
        self.snap_ratio = 0.0

    def solve(self, thermal_grid, optical_grid=None, now=None):
        """반환 dict:
            mask     : 인체 마스크 (표시용)
            profile  : 외곽선 프로파일 (표시용)
            raw      : 이번 프레임의 개방 거리 [mm]
            targets  : 평활을 거친 액츄에이터 목표 [mm]  <- 이걸 보냅니다
            infos    : 열별 진단

        시간 계산과 평활은 solve_mask() 가 맡는다. 여기서는 마스크만 만든다.
        """
        mask = body_mask(thermal_grid, self.tmin, self.tmax,
                         mm_per_px=self.mm_px, largest_only=self.largest_only,
                         relative=self.relative, bias=self.bias)
        return self.solve_mask(mask, optical_grid, now)

    def solve_mask(self, mask, optical_grid=None, now=None):
        """**이미 만들어진 인체 마스크**로 리프 목표를 구한다.

        광카메라 분할처럼 마스크를 밖에서 받아올 때 쓴다. solve() 는 열화상 온도에서
        마스크를 직접 만들지만, 여기서는 그 단계를 건너뛴다. 평활·변화율 제한 등
        나머지는 똑같이 적용하므로 두 경로의 거동이 갈리지 않는다.

        2026-08-31 에 외곽선을 광카메라에서 따오기로 방향을 바꾸면서 나눴다.
        열화상은 해상도가 160x120 이고 잔상·드리프트가 있어 외곽선을 따기에 불리하다.
        광카메라는 640x480 에 그런 문제가 없다. 필요한 것은 온도가 아니라 외곽선이다.
        """
        import time as _t
        now = _t.time() if now is None else now
        if self.last_t is None:
            dt = 0.1
        else:
            dt = max(1e-3, now - self.last_t)
        self.last_t = now

        prof = top_profile(mask)
        if self.fusion and optical_grid is not None:
            prof, self.snap_ratio = snap_to_edges(prof, edge_map(optical_grid), self.mm_px)
        else:
            self.snap_ratio = 0.0
        prof = smooth_profile(prof)

        raw, infos = openings_from_profile(prof, mask, self.h, self.w, self.mm_px,
                                           n_cols=self.n, margin_mm=self.margin)

        # --- 시간 평활 (비대칭) ---
        # 여는 방향(사람에게서 물러나는 쪽)은 즉시, 닫는 방향만 천천히.
        #
        # 대칭으로 평활하면 사람이 갑자기 움직였을 때 명령이 뒤늦게 따라가고,
        # 그 지연 동안 리프가 실제로 몸을 물고 있게 됩니다. 평활은 떨림을 줄이려는
        # 것이지 안전을 깎으라는 게 아니므로, 위험한 방향에만 겁니다.
        max_step = MAX_RATE_MM_S * dt
        for i in range(self.n):
            if raw[i] > self.smoothed[i]:
                self.smoothed[i] = raw[i]                 # 더 열어야 한다 -> 지체 없이
            else:
                ema = self.smoothed[i] + EMA_ALPHA * (raw[i] - self.smoothed[i])
                self.smoothed[i] += max(ema - self.smoothed[i], -max_step)

        targets = [to_actuator(v, invert=INVERT_LEAF) for v in self.smoothed]
        return {"mask": mask, "profile": prof, "raw": raw,
                "targets": targets, "infos": infos}

    def changed_enough(self, targets, deadband=SEND_DEADBAND_MM):
        """데드밴드보다 크게 움직였는가. 아두이노를 쓸데없이 흔들지 않기 위한 것."""
        if any(v is None for v in self.sent):
            return True
        return any(abs(t - s) >= deadband for t, s in zip(targets, self.sent))

    def mark_sent(self, targets):
        self.sent = list(targets)
