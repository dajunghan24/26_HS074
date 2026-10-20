/*
 * AMLA_8CH_75MM  —  AMLA 75MM 리니어 액츄에이터 제어 (8엽 MLC 리프 구동부)
 * ============================================================================
 * 2026-09-06. AMLA_4CH_100MM 에서 갈라져 나온 8채널판입니다.
 * 바뀐 것은 세 가지뿐이고, 프로토콜(시리얼 명령/응답)은 그대로라 mlc_driver.py 를
 * 고치지 않고 씁니다.
 *
 *   1) 액츄에이터가 행정 75mm 였습니다 (100mm 이 아니었음 — 2026-09-06 정정).
 *   2) BANK 명령 추가. 보드가 자기가 앞행인지 뒷행인지를 EEPROM 에 기억합니다.
 *   3) 뒷행(BANK R)은 IN 두 핀을 뒤집어 씁니다 — 그 행의 액츄에이터 리드가
 *      반대로 물려 있어서, 이렇게 해야 방향이 실물과 맞습니다.
 *
 * ── 8채널 구성 ──────────────────────────────────────────────────────────────
 *   UNO 는 여전히 한 대당 4채널입니다. '8CH' 은 UNO 두 대로 이루는 시스템 전체를
 *   가리킵니다. 두 대에 이 **같은 스케치 하나**를 올리고, BANK 로만 구분합니다.
 *
 *        왼쪽 ─────────────────────> 오른쪽
 *   앞행    1     2     3     4      BANK F   (/dev/mlc_uno2)
 *   뒷행    5     6     7     8      BANK R   (/dev/mlc_uno1)
 *
 *   ★ 포트 이름과 행이 반대입니다. mlc_uno1 이 뒷행입니다. 2026-09-06 에 이걸
 *     반대로 알고 한참 구동한 사고가 있었습니다. 그래서 BANK 를 넣었습니다.
 *
 * ── 하드웨어 ────────────────────────────────────────────────────────────────
 *   Arduino UNO  +  L298N(확장형, INA~IND / OUTA~OUTD 표기) x2  +  액츄에이터 x4
 *   액츄에이터 사양 : DC 12V, 행정 75mm, 추진/장력 20N, 홀딩토크 20N,
 *                     2선식(RED +, BLACK −), 양 끝단 리미트 스위치 내장.
 *
 * ── 결선 (2026-08-28 확정 — 사용자 실배선) ─────────────────────────────────
 *   ── 1번 모터 드라이버 ──
 *      1번 모터 : OUTA + OUTB      INA -> D2      INB -> D3
 *      2번 모터 : OUTC + OUTD      INC -> D4      IND -> D5
 *   ── 2번 모터 드라이버 ──
 *      3번 모터 : OUTA + OUTB      INA -> D6      INB -> D7
 *      4번 모터 : OUTC + OUTD      INC -> D8      IND -> D9
 *
 *   전원 : 12V 어댑터 (+) -> 두 드라이버의 VCC(+12V)
 *          12V 어댑터 (−) -> 두 드라이버의 GND
 *   GND  : 아두이노 GND ↔ 각 드라이버 GND 를 "각각 직접" 연결 (빵판 경유 금지).
 *          이게 빠지면 신호 기준이 없어 IN 이 전혀 먹지 않습니다.
 *   아두이노 전원은 USB. L298N 의 +5V 단자는 쓰지 않습니다.
 *
 *   ★★ ENA / ENB 점퍼캡은 4개 모두 "꽂아 둔" 상태여야 합니다. ★★
 *      이번 배선에는 EN 선이 없습니다. 캡이 EN 을 5V 로 잡아주는 구조이므로
 *      캡을 빼면 어느 채널도 절대 움직이지 않습니다. (설명서의 "ENA 점퍼 제거"는
 *      PWM 속도 제어를 할 때의 이야기이고, 이 스케치는 속도 제어를 하지 않습니다.)
 *      즉 속도는 항상 최대 고정이고, 정지는 EN 을 내리는 게 아니라
 *      IN=IN=LOW 로 합니다.
 *
 *   비어 있는 핀 : D10~D13, A1~A5 (D0/D1 은 시리얼, A0 은 진단 프로브)
 *
 * ── 위치 제어 방식 : 개루프입니다. 반드시 이해하고 쓰세요 ──────────────────
 *   이 액츄에이터에는 위치 피드백이 없습니다. 양 끝단 리미트 스위치만 내장.
 *   "완전 수축 = 0 mm" 를 원점으로 잡고, 그 뒤로는 경과 시간 ÷ 속도(ms/mm) 로
 *   위치를 추정합니다. 펌웨어는 모터가 실제로 돌았는지 알 수 없습니다.
 *   ※ 12V 어댑터를 안 꽂아도 시리얼은 멀쩡히 동작하고 "이동 완료" 까지 보고합니다.
 *      모터가 안 도는데 통신만 되면 배선을 뒤지기 전에 12V 부터 확인하세요.
 *   오차가 누적되므로 주기적으로 HOME 하세요.
 *
 * ── 프로토콜 (라즈베리파이 5 연동용) ──────────────────────────────────────
 *   115200 8N1, 한 줄 = 한 명령, 개행 \n. 명령은 대소문자 구분 안 함.
 *   모든 응답은 첫 토큰으로 종류가 구분됩니다:
 *     OK  ...  명령 수락
 *     ERR ...  거부 (사유 포함)
 *     ST  ...  상태 한 줄
 *     EV  ...  비동기 이벤트 (이동 완료 / 원점 완료 / 페일세이프)
 *     #   ...  사람이 읽는 설명. 파이썬은 '#' 로 시작하는 줄을 버리면 됩니다.
 *
 *   명령                       설명
 *   ------------------------   ------------------------------------------------
 *   PING                       살아있는지 확인
 *   HELP                       명령 목록
 *   CFG                        핀·보정값·한계 덤프
 *   STAT                       상태 1줄:  ST <p1> <p2> <p3> <p4> <busy> <homed>
 *   STREAM <ms>                STAT 자동 송출 주기. 0 이면 끔
 *   HOME [ch]                  원점 복귀. ch 생략 시 전 채널(시차 출발)
 *   GO <ch> <mm>               절대 위치 이동
 *   MOVE <ch> <dmm>            상대 이동 (+/-)
 *   SET <m1> <m2> <m3> <m4>    전 채널 목표 일괄 지정. 건너뛸 채널은 x
 *   STOP [ch]                  정지. ch 생략 시 전 채널
 *   JOG <ch> <E|R> <ms>        원점 없이 생으로 구동 (배선 확인용)
 *   TEST [ch]                  자가진단: 채널을 하나씩 뻗었다 넣음
 *   CAL <ch> <mm>              직전 이동의 실측값으로 속도 보정
 *   LIM <mm>                   최대 행정 설정
 *   WD <0|1>                   통신 감시 on/off
 *   FS <mm>                    통신 두절 시 대피할 '완전 차폐' 위치
 *   MAP / GND / VOLT           A0 프로브 진단 (아래 설명 참조)
 *
 *   ※ 이전 버전에 있던 EN <ch> <0|1> 명령은 삭제했습니다. EN 선이 없습니다.
 *
 *   전형적인 파이썬 사용 흐름:
 *     PING → CFG → TEST → HOME → (카메라 프레임마다) SET 12.0 30.5 8.0 22.0 → STAT
 *
 * ── 처음 올렸을 때 순서 ────────────────────────────────────────────────────
 *   1) 12V 어댑터 연결 확인, ENA/ENB 캡 4개 확인
 *   2) 시리얼 모니터 115200 → PING → CFG
 *   3) TEST : 채널을 하나씩 뻗었다 넣습니다. 눈으로 보며
 *      "CH n 이 몇 번 액츄에이터인가", "방향이 맞는가" 를 확정하세요.
 *      방향이 반대면 아래 CH_PINS 의 그 줄 두 숫자만 바꿔 적습니다. 예 {2,3}->{3,2}
 *   4) HOME → GO 1 30 같은 식으로 위치 이동
 *   5) 실측 이동량이 명령값과 다르면 CAL <ch> <실측mm> 로 보정하고,
 *      출력된 값을 아래 MS_EXT[] / MS_RET[] 에 옮겨 적어 영구 반영
 * ============================================================================
 */

#include <string.h>
#include <stdlib.h>
#include <EEPROM.h>

/* ========================== 사용자 설정 ========================== */

const uint8_t N_CH = 4;

struct PinSet { uint8_t in1, in2; };

/* IN1 이 HIGH 면 뻗음(extend), IN2 가 HIGH 면 들어옴(retract).
 * 어떤 채널이 반대로 돌면 그 줄의 두 숫자만 바꿔 적으세요. 예: {2,3} -> {3,2}
 * (모터 리드 RED/BLACK 을 다시 꽂을 때마다 방향이 뒤집힐 수 있습니다.) */
const PinSet CH_PINS[N_CH] = {
  /* CH1 */ { 2, 3 },   // 드라이버 1  INA/INB  -> OUTA+OUTB -> 1번 모터
  /* CH2 */ { 4, 5 },   // 드라이버 1  INC/IND  -> OUTC+OUTD -> 2번 모터
  /* CH3 */ { 6, 7 },   // 드라이버 2  INA/INB  -> OUTA+OUTB -> 3번 모터
  /* CH4 */ { 8, 9 },   // 드라이버 2  INC/IND  -> OUTC+OUTD -> 4번 모터
};

/* ── 앞행(F) 전용 IN 핀 교체 — 2026-09-07 ──────────────────────────────────
 * 0 = 변경 없음(위 CH_PINS 를 그대로 씀). 값이 있으면 그 채널만 갈아탄다.
 *
 * 앞행 2번 모터가 양방향 무반응이었고 **드라이버 LED 도 안 켜졌다.** LED 는
 * 모터보다 앞단이라, 신호가 출력단까지 못 갔다는 뜻이다. 같은 L298N 을 쓰는
 * 1번이 멀쩡했으므로 12V·GND·5V 는 무죄. 남은 용의자가 D4/D5 계통이라
 * 배선을 D10/D11 로 옮기고 여기서 맞춰 준다.
 *
 * ⚠️ **뒷행에는 적용하면 안 된다.** 뒷행의 CH2 는 6번 모터이고 그 배선은
 *    여전히 D4/D5 다. 뱅크를 안 가리면 6번이 같이 죽는다.
 *
 * 원인이 밝혀져 D4/D5 로 되돌리면 CH2 줄을 { 0, 0 } 으로. */
const PinSet CH_PINS_F[N_CH] = {
  /* CH1 */ { 0,  0  },
  /* CH2 */ { 10, 11 },   // D4/D5 계통 고장 의심으로 임시 이설
  /* CH3 */ { 0,  0  },
  /* CH4 */ { 0,  0  },
};

/* ================= 뱅크(행) 식별 — EEPROM 에 저장 =========================
 * UNO 두 대는 스케치도 CFG 도 완전히 같아서 소프트웨어로 구분할 수 없습니다.
 * 2026-09-06 에 앞뒤 행을 반대로 알고 한참 구동한 사고가 실제로 있었습니다.
 * 그래서 보드마다 "자기가 어느 행인지"를 EEPROM 에 한 번 적어 둡니다.
 *
 *   BANK F   앞행 (모터 1~4). CH_PINS 를 그대로 씁니다.
 *   BANK R   뒷행 (모터 5~8). CH_PINS 의 두 핀을 서로 바꿔 씁니다.
 *            뒷행은 액츄에이터 리드가 반대로 물려 있어, 이렇게 해야
 *            HOME(수축)·GO(돌출)·FS(실패안전) 방향이 모두 실물과 맞습니다.
 *            OUT 단자의 두 선을 바꿔 끼우는 것과 전기적으로 동일합니다.
 *
 * 보드당 한 번만 설정하면 전원을 빼도 남습니다:
 *   BANK R    뒷행 보드에 붙은 상태에서 한 번
 *   BANK      인자 없이 보내면 현재 값만 알려줍니다
 *
 * 부팅 배너·PING·CFG 에 뱅크가 찍히므로 **호스트가 포트와 행이 맞는지 검증**할
 * 수 있습니다. 그게 이 기능의 핵심 목적입니다. 미설정('?')이면 앞행과 똑같이
 * (뒤집지 않고) 동작하되, 배너에 UNSET 이라고 알립니다.
 */
const int     EE_ADDR_MAGIC = 0;
const int     EE_ADDR_BANK  = 1;
const uint8_t EE_MAGIC      = 0x4D;   /* 'M'. 다르면 미설정으로 봅니다 */

char gBank = '?';                     /* 'F' | 'R' | '?'(미설정) */

char bankLoad() {
  if (EEPROM.read(EE_ADDR_MAGIC) != EE_MAGIC) return '?';
  char b = (char)EEPROM.read(EE_ADDR_BANK);
  return (b == 'F' || b == 'R') ? b : '?';
}

void bankSave(char b) {
  EEPROM.update(EE_ADDR_MAGIC, EE_MAGIC);
  EEPROM.update(EE_ADDR_BANK, (uint8_t)b);
}

const __FlashStringHelper* bankName(char b) {
  if (b == 'F') return F("FRONT(1-4)");
  if (b == 'R') return F("REAR(5-8)");
  return F("UNSET");
}

/* 채널별 이동 속도 [ms/mm]. 2026-08-27 실측값을 옮겨온 것입니다.
 * 배선이 바뀌어도 모터·전압이 같으면 속도는 그대로지만,
 * 채널↔모터 대응이 달라졌을 수 있으니 TEST 로 확인한 뒤 CAL 로 다시 잡으세요.
 * ⚠️ 이 값들은 **앞행 보드에서만** 잰 것입니다. 뒷행(BANK R) 4채널은 한 번도
 *    실측한 적이 없고 이 숫자를 물려받았을 뿐입니다. 개루프라 이 값이 곧
 *    위치 정확도이므로, 뒷행은 CAL 로 따로 잡아야 합니다. */
float MS_EXT[N_CH] = { 77.90, 77.90, 77.90, 81.25 };   // 뻗을 때
float MS_RET[N_CH] = { 76.80, 82.03, 77.90, 81.15 };   // 들어올 때

/* 액츄에이터 자체 행정 75mm (2026-09-06 정정 — 그전까지 100mm 로 잘못 알고 있었음).
 * LIM 명령의 상한으로만 쓰입니다. 100 으로 두면 LIM 이 부품 행정을 넘는 값을
 * 받아들여, 도달할 수 없는 위치를 '도달했다'고 보고하게 됩니다. */
const float ACTUATOR_STROKE_MM = 75.0;

/* 실제로 쓰는 행정. MLC 설계상 60mm 만 씁니다.
 * 벤치에서 전 구간을 보고 싶으면 LIM 명령으로 올리세요. */
float MAX_TRAVEL_MM = 60.0;

/* 원점 복귀용 스위프 거리. MAX_TRAVEL 과 무관하게 항상 끝까지 밀어야 하므로
 * 실제 행정보다 넉넉하게 잡습니다. LIM 을 낮춰도 원점은 정확합니다.
 * 120mm = MAX_TRAVEL(60) 의 두 배이고 부품 행정 75mm 보다도 크므로, 어디서
 * 출발하든 확실히 끝단에 닿습니다(약 9초). 끝단에서는 내장 리미트 스위치가
 * 멈추므로 넉넉해도 무해합니다. 다만 스톨 상태를 길게 끌지는 마세요.
 * 행정이 75mm 로 바뀌어도 이 값은 그대로 둡니다 — 이미 충분히 큽니다. */
const float         HOMING_SWEEP_MM  = 120.0;
const unsigned long HOMING_EXTRA_MS  = 800;   // 스톱에 확실히 눌러주는 여유

/* 전 채널 원점 복귀 시 채널별 '출발 시차'.
 *
 * 한 채널이 원점을 잡는 데 약 10초(스위프 120mm)가 걸립니다. 시차를 250ms 로
 * 두면 사실상 네 채널이 통째로 겹쳐서 돕니다.
 *
 * 2026-08-27 실측: 12V 2A 어댑터로는 그 동시 구동을 감당하지 못했습니다.
 *     액츄에이터 1개  주행 0.5~1A   스톨 1.5~2.5A
 *     4개 동시        주행 2~4A     스톨 6~10A      <- 2A 정격의 3~5배
 *   전압이 주저앉아 모터가 딸각거리기만 하고, 채널마다 됐다 안 됐다 했습니다.
 *   배선 고장으로 착각하기 딱 좋은 증상입니다.
 *
 * 그래서 앞 채널이 끝난 뒤 다음이 출발하도록 시차를 스위프보다 길게 잡습니다.
 * 전체 원점 복귀가 약 45초로 늘어나는 대신 순간 전류가 1/4 로 떨어집니다.
 *
 * ★ 12V 5A 이상(권장 10A) 어댑터로 바꾸면 250 으로 되돌리세요. 훨씬 빠릅니다.
 *   되돌릴 때 호스트 쪽 home() 타임아웃도 같이 줄이면 됩니다. */
const unsigned long HOME_STAGGER_MS  = 11000;  // 순차 (2A 어댑터 대응). 전원 넉넉하면 250.

const unsigned long REVERSE_PAUSE_MS = 60;    // 방향 반전 전 멈춤(기어 보호)
const float         POS_DEADBAND_MM  = 0.3;   // 이보다 작은 차이는 무시

/* 통신 감시. WD 1 로 켜면 이 시간 동안 명령이 없을 때 전 채널을 FS 위치로.
 * FS 기본값 0.0 은 "완전 수축". 기구를 붙인 뒤 실제 차폐 위치로 FS 를 바꾸세요. */
const unsigned long LINK_TIMEOUT_MS = 3000;
float gFailsafeMm = 0.0;

/* 자가진단(TEST) 파라미터 */
const unsigned long TEST_JOG_MS   = 1500;
const unsigned long TEST_PAUSE_MS = 700;

/* 배선 진단 프로브 */
const uint8_t PIN_PROBE = A0;

/* ====================== 여기서부터 내부 구현 ====================== */

enum ChState : uint8_t { ST_IDLE = 0, ST_MOVING, ST_HOMING, ST_WAIT_HOME, ST_WAIT_MOVE };

struct Channel {
  uint8_t in1, in2;
  float   msExt, msRet;

  float   pos;            // 마지막으로 확정된 위치 [mm]
  bool    homed;

  uint8_t state;
  bool    dirExt;
  float   startPos;
  float   targetPos;
  unsigned long startMs;  // 대기 상태에서는 '시작 예정 시각'
  unsigned long endMs;
  float   pendingTarget;

  bool          lastValid;   // CAL 이 참조할 직전 이동 기록
  bool          lastExt;
  float         lastStart;
  unsigned long lastMs;
};

Channel ch[N_CH];

bool          gEvents     = true;    // EV 줄 송출
bool          gWatchdog   = false;
unsigned long gLastCmdMs  = 0;
bool          gFsLatched  = false;
unsigned long gStreamMs   = 0;       // 0 = STAT 자동 송출 끔
unsigned long gStreamLast = 0;

static char    rxBuf[72];
static uint8_t rxLen = 0;

/* millis() 오버플로에 안전한 시간 비교 */
static inline bool timeReached(unsigned long t) { return (long)(millis() - t) >= 0; }

/* ---------------- 모터 저수준 구동 ----------------
 * EN 은 점퍼캡으로 5V 고정이므로 여기서 건드릴 것이 없습니다.
 * IN1=IN2=LOW  -> 프리런 정지 (이 스케치의 정지)
 * IN1=IN2=HIGH -> 브레이크. 쓰지 않습니다.
 */
void drvExtend (uint8_t i) { digitalWrite(ch[i].in1, HIGH); digitalWrite(ch[i].in2, LOW ); }
void drvRetract(uint8_t i) { digitalWrite(ch[i].in1, LOW ); digitalWrite(ch[i].in2, HIGH); }
void drvStop   (uint8_t i) { digitalWrite(ch[i].in1, LOW ); digitalWrite(ch[i].in2, LOW ); }

/* 뱅크에 맞춰 IN 두 핀의 역할을 정하고 핀을 초기화합니다.
 * 'R'(뒷행)이면 두 핀을 서로 바꿉니다 = 모터 리드를 바꿔 끼운 것과 같습니다.
 * setup() 과 BANK 명령에서 모두 이 함수를 씁니다. */
void applyBankPins() {
  const bool flip = (gBank == 'R');
  for (uint8_t i = 0; i < N_CH; i++) {
    /* 앞행 전용 교체를 먼저 반영한다. 0 이면 기본 배정을 그대로 쓴다.
     * 뒷행(flip)은 절대 건드리지 않는다 — 그 쪽 배선은 안 바뀌었다. */
    PinSet p = CH_PINS[i];
    if (!flip && CH_PINS_F[i].in1 != 0) p = CH_PINS_F[i];
    ch[i].in1 = flip ? p.in2 : p.in1;
    ch[i].in2 = flip ? p.in1 : p.in2;
    pinMode(ch[i].in1, OUTPUT);
    pinMode(ch[i].in2, OUTPUT);
    drvStop(i);
  }
}

/* ---------------- 위치 추정 ----------------
 * 이동 중에도 경과 시간으로 현재 위치를 역산합니다.
 * 이동 도중에 목표를 바꿔치기(선점)할 수 있는 근거가 이 함수입니다.
 */
float currentPos(uint8_t i) {
  if (ch[i].state != ST_MOVING) return ch[i].pos;

  float rate = ch[i].dirExt ? ch[i].msExt : ch[i].msRet;
  if (rate <= 0.0) return ch[i].pos;

  float travelled = (float)(millis() - ch[i].startMs) / rate;
  float p = ch[i].dirExt ? (ch[i].startPos + travelled)
                         : (ch[i].startPos - travelled);

  if ( ch[i].dirExt && p > ch[i].targetPos) p = ch[i].targetPos;
  if (!ch[i].dirExt && p < ch[i].targetPos) p = ch[i].targetPos;
  if (p < 0.0)           p = 0.0;
  if (p > MAX_TRAVEL_MM) p = MAX_TRAVEL_MM;
  return p;
}

bool isBusy(uint8_t i) { return ch[i].state != ST_IDLE; }

/* ---------------- 정지 ---------------- */

void chStop(uint8_t i) {
  if (ch[i].state == ST_MOVING) ch[i].pos = currentPos(i);
  drvStop(i);
  ch[i].state = ST_IDLE;
}

void stopAll() { for (uint8_t i = 0; i < N_CH; i++) chStop(i); }

/* 생으로 구동한 뒤에는 추정 위치를 믿을 수 없으므로 원점을 무효화합니다 */
void invalidateAll() {
  for (uint8_t i = 0; i < N_CH; i++) {
    drvStop(i);
    ch[i].pos = 0.0; ch[i].homed = false;
    ch[i].state = ST_IDLE; ch[i].lastValid = false;
  }
  gLastCmdMs = millis();
}

/* ---------------- 이동 (논블로킹) ---------------- */

void beginSegment(uint8_t i, float target, bool ext) {
  float now  = currentPos(i);
  float dist = fabs(target - now);
  unsigned long ms = (unsigned long)(dist * (ext ? ch[i].msExt : ch[i].msRet));

  ch[i].startPos  = now;
  ch[i].targetPos = target;
  ch[i].dirExt    = ext;
  ch[i].startMs   = millis();
  ch[i].endMs     = ch[i].startMs + ms;
  ch[i].state     = ST_MOVING;

  ch[i].lastValid = true;
  ch[i].lastExt   = ext;
  ch[i].lastStart = now;
  ch[i].lastMs    = ms;

  if (ext) drvExtend(i); else drvRetract(i);
}

/* 성공하면 true. 실패 사유는 호출부가 ERR 로 알립니다. */
bool chMoveTo(uint8_t i, float targetMm) {
  if (ch[i].state == ST_HOMING || ch[i].state == ST_WAIT_HOME) return false;
  if (!ch[i].homed) return false;

  if (targetMm < 0.0)           targetMm = 0.0;
  if (targetMm > MAX_TRAVEL_MM) targetMm = MAX_TRAVEL_MM;

  float now   = currentPos(i);
  float delta = targetMm - now;

  if (fabs(delta) < POS_DEADBAND_MM) {           // 이미 목표 근처
    if (ch[i].state == ST_MOVING) { ch[i].pos = now; drvStop(i); ch[i].state = ST_IDLE; }
    return true;
  }

  bool ext = (delta > 0.0);

  /* 진행 중인데 방향이 반대면 잠깐 멈췄다 반전 (기어 보호) */
  if (ch[i].state == ST_MOVING && ext != ch[i].dirExt) {
    ch[i].pos           = now;
    drvStop(i);
    ch[i].pendingTarget = targetMm;
    ch[i].dirExt        = ext;
    ch[i].startMs       = millis() + REVERSE_PAUSE_MS;
    ch[i].state         = ST_WAIT_MOVE;
    return true;
  }

  if (ch[i].state == ST_WAIT_MOVE) { ch[i].pendingTarget = targetMm; return true; }

  beginSegment(i, targetMm, ext);
  return true;
}

/* ---------------- 원점 복귀 (논블로킹) ---------------- */

void chHome(uint8_t i, unsigned long delayMs) {
  drvStop(i);
  ch[i].state     = ST_WAIT_HOME;
  ch[i].startMs   = millis() + delayMs;
  ch[i].lastValid = false;
}

void beginHome(uint8_t i) {
  drvRetract(i);
  ch[i].startMs = millis();
  ch[i].endMs   = ch[i].startMs
                + (unsigned long)(HOMING_SWEEP_MM * ch[i].msRet)
                + HOMING_EXTRA_MS;
  ch[i].state   = ST_HOMING;
}

/* 돌입 전류가 겹치지 않게 채널마다 시차를 두고 출발시킵니다 */
void homeAll() {
  for (uint8_t i = 0; i < N_CH; i++) chHome(i, (unsigned long)i * HOME_STAGGER_MS);
}

/* ---------------- 매 loop 갱신 ---------------- */

void updateMotors() {
  for (uint8_t i = 0; i < N_CH; i++) {
    switch (ch[i].state) {

      case ST_MOVING:
        if (timeReached(ch[i].endMs)) {
          drvStop(i);
          ch[i].pos   = ch[i].targetPos;
          ch[i].state = ST_IDLE;
          if (gEvents) {
            Serial.print(F("EV DONE ")); Serial.print(i + 1);
            Serial.print(' ');           Serial.println(ch[i].pos, 1);
          }
        }
        break;

      case ST_WAIT_MOVE:
        if (timeReached(ch[i].startMs)) {
          float t = ch[i].pendingTarget;
          float d = t - ch[i].pos;
          if (fabs(d) < POS_DEADBAND_MM) ch[i].state = ST_IDLE;
          else                           beginSegment(i, t, d > 0.0);
        }
        break;

      case ST_WAIT_HOME:
        if (timeReached(ch[i].startMs)) beginHome(i);
        break;

      case ST_HOMING:
        if (timeReached(ch[i].endMs)) {
          drvStop(i);
          ch[i].pos   = 0.0;
          ch[i].homed = true;
          ch[i].state = ST_IDLE;
          if (gEvents) { Serial.print(F("EV HOMED ")); Serial.println(i + 1); }
        }
        break;

      default: break;
    }
  }
}

/* ---------------- 통신 감시 ----------------
 * 파이가 죽거나 USB 가 빠졌을 때 리프가 열린 채 방치되지 않도록
 * 전 채널을 FS 위치(완전 차폐)로 보냅니다. 안전한 쪽으로 실패시키는 장치입니다.
 */
void updateWatchdog() {
  if (!gWatchdog) return;

  if (millis() - gLastCmdMs > LINK_TIMEOUT_MS) {
    if (!gFsLatched) {
      gFsLatched = true;
      Serial.print(F("EV FAILSAFE ")); Serial.println(gFailsafeMm, 1);
      for (uint8_t i = 0; i < N_CH; i++) if (ch[i].homed) chMoveTo(i, gFailsafeMm);
    }
  } else {
    gFsLatched = false;
  }
}

/* ---------------- 출력 ---------------- */

/* ST <p1> <p2> <p3> <p4> <busy비트열> <homed비트열> */
void printStat() {
  Serial.print(F("ST"));
  for (uint8_t i = 0; i < N_CH; i++) { Serial.print(' '); Serial.print(currentPos(i), 1); }
  Serial.print(' ');
  for (uint8_t i = 0; i < N_CH; i++) Serial.print(isBusy(i)   ? '1' : '0');
  Serial.print(' ');
  for (uint8_t i = 0; i < N_CH; i++) Serial.print(ch[i].homed ? '1' : '0');
  Serial.println();
}

void printCfg() {
  Serial.println(F("# --- config (AMLA_8CH_75MM) ---"));
  Serial.print(F("# bank ")); Serial.print(gBank);
  Serial.print(F(" = "));     Serial.print(bankName(gBank));
  Serial.println(gBank == 'R' ? F("  [IN pins swapped]") : F(""));
  for (uint8_t i = 0; i < N_CH; i++) {
    Serial.print(F("# CH")); Serial.print(i + 1);
    Serial.print(F("  IN1=D")); Serial.print(ch[i].in1);
    Serial.print(F("  IN2=D")); Serial.print(ch[i].in2);
    Serial.print(F("  ext "));  Serial.print(ch[i].msExt, 2);
    Serial.print(F("  ret "));  Serial.print(ch[i].msRet, 2);
    Serial.println(F(" ms/mm"));
  }
  Serial.println(F("# EN: jumper caps -> 5V (no PWM, no EN wire)"));
  Serial.print(F("# travel 0..")); Serial.print(MAX_TRAVEL_MM, 1);
  Serial.print(F(" mm  of stroke ")); Serial.print(ACTUATOR_STROKE_MM, 1);
  Serial.println(F(" mm"));
  Serial.print(F("# failsafe ")); Serial.print(gFailsafeMm, 1);
  Serial.print(F(" mm   watchdog ")); Serial.println(gWatchdog ? F("on") : F("off"));
  Serial.print(F("# stream "));   Serial.print(gStreamMs); Serial.println(F(" ms"));
  Serial.println(F("OK CFG"));
}

void printHelp() {
  Serial.println(F("# AMLA_8CH_75MM"));
  Serial.println(F("# PING | HELP | CFG | STAT | STREAM <ms>"));
  Serial.println(F("# HOME [ch] | GO <ch> <mm> | MOVE <ch> <dmm> | SET <m1..m4>"));
  Serial.println(F("# STOP [ch] | JOG <ch> <E|R> <ms> | TEST [ch] | CAL <ch> <mm>"));
  Serial.println(F("# LIM <mm> | WD <0|1> | FS <mm> | BANK [F|R]"));
  Serial.println(F("# MAP | GND | VOLT   (A0 probe diagnostics)"));
  Serial.println(F("OK HELP"));
}

/* ---------------- 배선 진단 (A0 프로브) ----------------
 * 점퍼선 한 가닥을 A0 에 꽂고 반대쪽 끝으로 로직 핀을 짚습니다.
 * !! +12V 단자와 OUT 단자는 절대 짚지 마세요. 아두이노가 즉시 망가집니다.
 *    짚어도 되는 곳은 IN·EN·5V·GND 뿐입니다. !!
 * 이 세 명령은 블로킹입니다. 벤치 전용이며 파이 운용 중에는 쓰지 마세요.
 */
float probeVolts() {
  analogRead(PIN_PROBE);                 // 멀티플렉서 안정화용 더미 읽기
  return analogRead(PIN_PROBE) * (5.0 / 1023.0);
}

/* ms 동안 대기하며 중단 요청(아무 입력)을 감시. 중단되면 false */
bool waitAbortable(unsigned long ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) {
    if (Serial.available()) { while (Serial.available()) Serial.read(); return false; }
  }
  return true;
}

/* MAP : 프로브를 댄 입력 핀이 아두이노 몇 번에 물려 있는지 자동으로 찾아줍니다.
 * 이번 배선은 IN 이 D2~D9 로 딱 이 범위이고 EN 선이 없으므로 오탐이 없습니다. */
void diagMap() {
  stopAll();
  Serial.println(F("# MAP: hold the A0 probe on ONE L298N input pin."));
  Serial.println(F("# driving D2..D9 HIGH one at a time. send anything to abort."));

  uint8_t hitPin = 0, hits = 0;

  for (uint8_t pin = 2; pin <= 9; pin++) {
    for (uint8_t q = 2; q <= 9; q++) digitalWrite(q, LOW);
    if (!waitAbortable(250)) { Serial.println(F("ERR MAP aborted")); invalidateAll(); return; }

    digitalWrite(pin, HIGH);
    if (!waitAbortable(400)) { digitalWrite(pin, LOW); Serial.println(F("ERR MAP aborted")); invalidateAll(); return; }

    float v = probeVolts();
    Serial.print(F("#   D")); Serial.print(pin);
    Serial.print(F(" HIGH -> A0 = ")); Serial.print(v, 2); Serial.print(F(" V"));
    if (v > 4.0) { Serial.println(F("   <<< CONNECTED")); hitPin = pin; hits++; }
    else           Serial.println();

    digitalWrite(pin, LOW);
    if (!waitAbortable(150)) { Serial.println(F("ERR MAP aborted")); invalidateAll(); return; }
  }

  for (uint8_t q = 2; q <= 9; q++) digitalWrite(q, LOW);
  invalidateAll();

  if (hits == 1) {
    Serial.print(F("OK MAP D")); Serial.println(hitPin);
  } else if (hits == 0) {
    Serial.println(F("# no Arduino pin reached the probe."));
    Serial.println(F("# -> jumper broken, not seated, or on the wrong pin."));
    Serial.println(F("OK MAP NONE"));
  } else {
    Serial.println(F("# several pins responded: inputs shorted, probe touching"));
    Serial.println(F("# two pins, or you probed an EN pin held at 5V (that is normal)."));
    Serial.print(F("OK MAP MULTI ")); Serial.println(hits);
  }
}

/* GND : A0 내부 풀업을 켜고 짚은 지점이 아두이노 GND 와 이어져 있는지 봅니다 */
void diagGnd() {
  stopAll();
  Serial.println(F("# GND: touch the probe to a GND point on the driver board."));
  Serial.println(F("#  <1V = CONNECTED (good)   >3V = NOT connected (this is the fault)"));

  pinMode(PIN_PROBE, INPUT_PULLUP);
  for (uint8_t k = 0; k < 40; k++) {                 // 20초
    if (!waitAbortable(500)) break;
    float v = probeVolts();
    Serial.print(F("#   A0 = ")); Serial.print(v, 2); Serial.print(F(" V  "));
    if      (v < 1.0) Serial.println(F("CONNECTED"));
    else if (v > 3.0) Serial.println(F("NOT connected"));
    else              Serial.println(F("??"));
  }
  pinMode(PIN_PROBE, INPUT);
  Serial.println(F("OK GND"));
}

/* VOLT : 그냥 전압계 */
void diagVolt() {
  stopAll();
  Serial.println(F("# VOLT: A0 voltmeter. logic pins only - never +12V or OUT."));
  for (uint8_t k = 0; k < 40; k++) {                 // 20초
    if (!waitAbortable(500)) break;
    float v = probeVolts();
    Serial.print(F("#   A0 = ")); Serial.print(v, 2); Serial.print(F(" V  "));
    if      (v < 0.8) Serial.println(F("LOW"));
    else if (v > 4.0) Serial.println(F("HIGH"));
    else              Serial.println(F("floating/mid"));
  }
  Serial.println(F("OK VOLT"));
}

/* ---------------- 자가진단 ----------------
 * 위치 제어를 거치지 않고 채널을 직접 구동합니다. 원점도 보정값도 불필요.
 * "이 채널에 물린 모터가 실제로 도는가" 와 "몇 번 액츄에이터인가" 만 봅니다.
 */
void selfTest(int only) {          // only < 0 이면 전 채널
  stopAll();
  Serial.println(F("# TEST: watch the actuators and note which label moves."));

  for (uint8_t i = 0; i < N_CH; i++) {
    if (only >= 0 && (int)i != only) continue;

    Serial.print(F("# CH")); Serial.print(i + 1);
    Serial.print(F(" (IN1=D")); Serial.print(ch[i].in1);
    Serial.print(F(" IN2=D"));  Serial.print(ch[i].in2);
    Serial.println(F(")  EXTEND >>>"));
    drvExtend(i);
    if (!waitAbortable(TEST_JOG_MS))   { drvStop(i); Serial.println(F("ERR TEST aborted")); invalidateAll(); return; }
    drvStop(i);
    if (!waitAbortable(TEST_PAUSE_MS)) { Serial.println(F("ERR TEST aborted")); invalidateAll(); return; }

    Serial.print(F("# CH")); Serial.print(i + 1); Serial.println(F("  <<< RETRACT"));
    drvRetract(i);
    if (!waitAbortable(TEST_JOG_MS))   { drvStop(i); Serial.println(F("ERR TEST aborted")); invalidateAll(); return; }
    drvStop(i);
    if (!waitAbortable(TEST_PAUSE_MS)) { Serial.println(F("ERR TEST aborted")); invalidateAll(); return; }
  }

  invalidateAll();
  Serial.println(F("# positions are UNKNOWN now - run HOME before GO/MOVE/SET."));
  Serial.println(F("OK TEST"));
}

/* ---------------- 보정 ---------------- */

void chCalibrate(uint8_t i, float actualMm) {
  if (isBusy(i))        { Serial.println(F("ERR CAL busy")); return; }
  if (!ch[i].lastValid) { Serial.println(F("ERR CAL no previous move")); return; }
  if (actualMm <= 0.2)  { Serial.println(F("ERR CAL distance too small")); return; }

  float newRate = (float)ch[i].lastMs / actualMm;
  if (ch[i].lastExt) ch[i].msExt = newRate; else ch[i].msRet = newRate;

  ch[i].pos = ch[i].lastStart + (ch[i].lastExt ? actualMm : -actualMm);
  if (ch[i].pos < 0.0)           ch[i].pos = 0.0;
  if (ch[i].pos > MAX_TRAVEL_MM) ch[i].pos = MAX_TRAVEL_MM;
  ch[i].lastValid = false;

  Serial.print(F("OK CAL ")); Serial.print(i + 1);
  Serial.print(ch[i].lastExt ? F(" EXT ") : F(" RET "));
  Serial.println(newRate, 2);
  Serial.println(F("# copy this into MS_EXT[]/MS_RET[] so it survives a reset."));
}

/* ---------------- 명령 해석 ---------------- */

uint8_t tokenize(char *s, char *tok[], uint8_t maxTok) {
  uint8_t n = 0;
  while (*s && n < maxTok) {
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) break;
    tok[n++] = s;
    while (*s && *s != ' ' && *s != '\t') s++;
    if (*s) *s++ = 0;
  }
  return n;
}

/* 1..N_CH 를 받아 0-based 인덱스로. 범위를 벗어나면 -1 */
int parseCh(const char *s) {
  int n = atoi(s);
  if (n < 1 || n > (int)N_CH) return -1;
  return n - 1;
}

void handleLine(char *line) {
  char *tok[6];
  uint8_t nt = tokenize(line, tok, 6);
  if (nt == 0) return;

  gLastCmdMs = millis();
  gFsLatched = false;

  const char *c = tok[0];

  if (!strcasecmp(c, "PING")) {
    /* 뱅크를 같이 실어 보냅니다. 호스트가 포트와 행이 맞는지 검증하는 근거입니다. */
    Serial.print(F("OK PING AMLA_8CH_75MM ")); Serial.println(gBank);
    return;
  }

  /* BANK          현재 뱅크만 알려줍니다
   * BANK F | R    EEPROM 에 저장하고 즉시 적용합니다 (뒷행이면 IN 핀을 뒤집습니다).
   *               방향이 바뀌므로 원점·위치 기억은 버립니다 — 다시 HOME 해야 합니다. */
  if (!strcasecmp(c, "BANK")) {
    if (nt >= 2) {
      char b = tok[1][0];
      if (b >= 'a' && b <= 'z') b -= 32;
      if (b != 'F' && b != 'R') { Serial.println(F("ERR BANK needs F or R")); return; }
      stopAll();
      bankSave(b);
      gBank = b;
      applyBankPins();
      invalidateAll();
      Serial.print(F("OK BANK ")); Serial.print(gBank);
      Serial.print(' ');           Serial.print(bankName(gBank));
      Serial.println(F(" - direction changed, re-HOME required"));
      return;
    }
    Serial.print(F("OK BANK ")); Serial.print(gBank);
    Serial.print(' ');           Serial.println(bankName(gBank));
    return;
  }
  if (!strcasecmp(c, "HELP") || !strcmp(c, "?")) { printHelp(); return; }
  if (!strcasecmp(c, "CFG"))  { printCfg();  return; }
  if (!strcasecmp(c, "STAT")) { printStat(); return; }

  if (!strcasecmp(c, "STREAM")) {
    if (nt < 2) { Serial.println(F("ERR STREAM needs ms")); return; }
    long v = atol(tok[1]);
    if (v < 0) v = 0;
    if (v > 0 && v < 50) v = 50;              // 너무 잦으면 시리얼이 막힙니다
    gStreamMs = (unsigned long)v;
    Serial.print(F("OK STREAM ")); Serial.println(gStreamMs);
    return;
  }

  if (!strcasecmp(c, "STOP")) {
    if (nt >= 2) {
      int i = parseCh(tok[1]);
      if (i < 0) { Serial.println(F("ERR STOP bad channel")); return; }
      chStop((uint8_t)i);
      Serial.print(F("OK STOP ")); Serial.println(i + 1);
    } else {
      stopAll();
      Serial.println(F("OK STOP"));
    }
    return;
  }

  if (!strcasecmp(c, "HOME")) {
    if (nt >= 2) {
      int i = parseCh(tok[1]);
      if (i < 0) { Serial.println(F("ERR HOME bad channel")); return; }
      chHome((uint8_t)i, 0);
      Serial.print(F("OK HOME ")); Serial.println(i + 1);
    } else {
      homeAll();
      Serial.println(F("OK HOME"));
    }
    return;
  }

  if (!strcasecmp(c, "GO") || !strcasecmp(c, "MOVE")) {
    if (nt < 3) { Serial.println(F("ERR needs <ch> <mm>")); return; }
    int i = parseCh(tok[1]);
    if (i < 0) { Serial.println(F("ERR bad channel")); return; }
    if (!ch[i].homed) { Serial.print(F("ERR not homed ")); Serial.println(i + 1); return; }

    float v = atof(tok[2]);
    float target = (!strcasecmp(c, "MOVE")) ? currentPos((uint8_t)i) + v : v;

    if (!chMoveTo((uint8_t)i, target)) { Serial.println(F("ERR move refused")); return; }
    Serial.print(F("OK ")); Serial.print(c);
    Serial.print(' ');      Serial.print(i + 1);
    Serial.print(' ');      Serial.println(target, 1);
    return;
  }

  /* SET <m1> <m2> <m3> <m4>  — 카메라 프레임마다 파이가 보내는 주력 명령.
   * 건너뛸 채널은 x 로 표기합니다. 예: SET 12.0 x 8.0 22.0 */
  if (!strcasecmp(c, "SET")) {
    uint8_t applied = 0, refused = 0;
    for (uint8_t i = 0; i < N_CH && (i + 1) < nt; i++) {
      const char *t = tok[i + 1];
      if (t[0] == 'x' || t[0] == 'X' || t[0] == '-') continue;
      if (!ch[i].homed) { refused++; continue; }
      if (chMoveTo(i, atof(t))) applied++; else refused++;
    }
    Serial.print(F("OK SET ")); Serial.print(applied);
    if (refused) { Serial.print(F(" refused ")); Serial.print(refused); }
    Serial.println();
    return;
  }

  /* JOG <ch> <E|R> <ms> — 원점 없이 생으로 구동. 배선 확인용. */
  if (!strcasecmp(c, "JOG")) {
    if (nt < 4) { Serial.println(F("ERR JOG needs <ch> <E|R> <ms>")); return; }
    int i = parseCh(tok[1]);
    if (i < 0) { Serial.println(F("ERR JOG bad channel")); return; }
    bool ext = (tok[2][0] == 'e' || tok[2][0] == 'E');
    long ms  = atol(tok[3]);
    if (ms < 1)     ms = 1;
    if (ms > 10000) ms = 10000;              // 안전 상한

    stopAll();
    Serial.print(F("# JOG CH")); Serial.print(i + 1);
    Serial.print(ext ? F(" EXTEND ") : F(" RETRACT "));
    Serial.print(ms); Serial.println(F(" ms"));

    if (ext) drvExtend((uint8_t)i); else drvRetract((uint8_t)i);
    bool done = waitAbortable((unsigned long)ms);
    drvStop((uint8_t)i);
    invalidateAll();
    Serial.println(done ? F("OK JOG") : F("ERR JOG aborted"));
    return;
  }

  if (!strcasecmp(c, "TEST")) {
    int only = -1;
    if (nt >= 2) {
      only = parseCh(tok[1]);
      if (only < 0) { Serial.println(F("ERR TEST bad channel")); return; }
    }
    selfTest(only);
    return;
  }

  if (!strcasecmp(c, "CAL")) {
    if (nt < 3) { Serial.println(F("ERR CAL needs <ch> <mm>")); return; }
    int i = parseCh(tok[1]);
    if (i < 0) { Serial.println(F("ERR CAL bad channel")); return; }
    chCalibrate((uint8_t)i, atof(tok[2]));
    return;
  }

  if (!strcasecmp(c, "LIM")) {
    if (nt < 2) { Serial.println(F("ERR LIM needs mm")); return; }
    float v = atof(tok[1]);
    if (v < 1.0)                  v = 1.0;
    if (v > ACTUATOR_STROKE_MM)   v = ACTUATOR_STROKE_MM;   // 부품 행정 75mm
    MAX_TRAVEL_MM = v;
    Serial.print(F("OK LIM ")); Serial.println(MAX_TRAVEL_MM, 1);
    return;
  }

  if (!strcasecmp(c, "WD")) {
    if (nt < 2) { Serial.println(F("ERR WD needs 0|1")); return; }
    gWatchdog  = (tok[1][0] != '0');
    gFsLatched = false;
    Serial.print(F("OK WD ")); Serial.println(gWatchdog ? 1 : 0);
    return;
  }

  if (!strcasecmp(c, "FS")) {
    if (nt < 2) { Serial.println(F("ERR FS needs mm")); return; }
    float v = atof(tok[1]);
    if (v < 0.0)           v = 0.0;
    if (v > MAX_TRAVEL_MM) v = MAX_TRAVEL_MM;
    gFailsafeMm = v;
    Serial.print(F("OK FS ")); Serial.println(gFailsafeMm, 1);
    return;
  }

  if (!strcasecmp(c, "MAP"))  { diagMap();  return; }
  if (!strcasecmp(c, "GND"))  { diagGnd();  return; }
  if (!strcasecmp(c, "VOLT")) { diagVolt(); return; }

  /* 예전 스케치에 있던 EN 명령. 이번 배선에는 EN 선이 없으므로 안내만 합니다. */
  if (!strcasecmp(c, "EN")) {
    Serial.println(F("ERR EN removed - ENA/ENB are tied to 5V by jumper caps"));
    return;
  }

  Serial.print(F("ERR unknown ")); Serial.println(c);
}

/* ---------------- setup / loop ---------------- */

void setup() {
  gBank = bankLoad();                 /* 핀 배정보다 먼저 읽어야 합니다 */

  for (uint8_t i = 0; i < N_CH; i++) {
    ch[i].msExt = MS_EXT[i];
    ch[i].msRet = MS_RET[i];

    ch[i].pos           = 0.0;
    ch[i].homed         = false;
    ch[i].state         = ST_IDLE;
    ch[i].dirExt        = true;
    ch[i].startPos      = 0.0;
    ch[i].targetPos     = 0.0;
    ch[i].pendingTarget = 0.0;
    ch[i].startMs       = 0;
    ch[i].endMs         = 0;
    ch[i].lastValid     = false;
    ch[i].lastExt       = true;
    ch[i].lastStart     = 0.0;
    ch[i].lastMs        = 0;
  }

  applyBankPins();                    /* in1/in2 배정 + pinMode + 정지 */

  Serial.begin(115200);
  while (!Serial) { ; }
  gLastCmdMs = millis();

  Serial.println();
  Serial.println(F("# AMLA_8CH_75MM ready. HELP for commands."));
  Serial.println(F("# wiring: IN D2..D9 (drv1 CH1/CH2, drv2 CH3/CH4), ENA/ENB = 5V caps."));
  Serial.print  (F("# bank: ")); Serial.print(gBank);
  Serial.print  (F(" = "));      Serial.print(bankName(gBank));
  if (gBank == 'R') Serial.print(F("  [IN pins swapped]"));
  Serial.println();
  if (gBank == '?') Serial.println(F("# !! bank not set - send BANK F or BANK R once."));
  Serial.println(F("# not homed yet - send HOME before GO/MOVE/SET."));
  Serial.println(F("OK BOOT"));
}

void loop() {
  /* 1) 시리얼 읽기 — 이동 중에도 매 순간 실행됩니다 */
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0) { rxBuf[rxLen] = 0; handleLine(rxBuf); rxLen = 0; }
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    }
    /* 버퍼가 넘치면 그 줄은 잘립니다. 명령은 모두 짧으므로 문제되지 않습니다. */
  }

  /* 2) 채널 상태 갱신 (delay() 를 대신하는 부분) */
  updateMotors();

  /* 3) 통신 두절 감시 */
  updateWatchdog();

  /* 4) 상태 자동 송출 */
  if (gStreamMs && millis() - gStreamLast >= gStreamMs) {
    gStreamLast = millis();
    printStat();
  }
}
