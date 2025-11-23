#!/usr/bin/env python3
import time
import numpy as np
import mujoco
from mujoco import viewer
import serial

XML_PATH = "hexapod_hardware.xml"

# ⚠️ 여기를 실제 포트 이름으로 바꿔줘
SERIAL_PORT = "/dev/tty.usbserial-0001"
BAUD_RATE   = 115200

SEND_HZ = 30.0  # ESP로 보내는 주기 (30Hz 정도면 충분)

# ---------- MuJoCo 모델 로드 ----------
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# keyframe "home"으로 초기화
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id == -1:
    raise RuntimeError("Keyframe 'home' not found")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# 액추에이터 이름 순서 (ESP32 쪽 18서보 순서와 반드시 동일!)
ACTUATOR_NAMES = [
    "left_rear_yaw_act",  "left_rear_hip_act",  "left_rear_knee_act",
    "right_rear_yaw_act", "right_rear_hip_act", "right_rear_knee_act",
    "left_mid_yaw_act",   "left_mid_hip_act",   "left_mid_knee_act",
    "right_mid_yaw_act",  "right_mid_hip_act",  "right_mid_knee_act",
    "left_front_yaw_act", "left_front_hip_act", "left_front_knee_act",
    "right_front_yaw_act","right_front_hip_act","right_front_knee_act",
]

actuator_ids = []
for name in ACTUATOR_NAMES:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        raise RuntimeError(f"Actuator '{name}' not found in model")
    actuator_ids.append(aid)
actuator_ids = np.array(actuator_ids, dtype=int)

# ---------- 시리얼 포트 오픈 ----------
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.001)
time.sleep(2.0)          # ESP 리셋 후 부팅 로그 쏟아지는 시간
ser.reset_input_buffer() # 초기 쓰레기 로그 제거

# ---------- ESP → 파이썬으로 받을 데이터 버퍼 ----------
NUM_LEGS = 6
last_fsr_norm   = np.zeros(NUM_LEGS, dtype=float)  # F 라인 (정규화 FSR 값)
last_leg_contact = np.zeros(NUM_LEGS, dtype=int)   # C 라인 (0/1 접촉 상태)


def ctrl_to_normalized():
    """
    MuJoCo data.ctrl (각 actuator ctrl 값)를 [-1, 1]로 정규화.
    ctrlrange: (min, max) 기준.
    """
    norms = np.zeros(len(actuator_ids), dtype=float)
    for i, aid in enumerate(actuator_ids):
        c_val = float(data.ctrl[aid])
        c_min, c_max = model.actuator_ctrlrange[aid]
        # min-max 정규화 → [-1,1]
        if c_max == c_min:
            norms[i] = 0.0
        else:
            t = (c_val - c_min) / (c_max - c_min)  # 0~1
            norms[i] = 2.0 * t - 1.0
        # 혹시 범위 넘으면 클립
        norms[i] = max(-1.0, min(1.0, norms[i]))
    return norms


def send_to_esp(norms: np.ndarray):
    """
    [-1,1] float 18개를 [-1000,1000] int로 스케일 후
    "S v0 v1 ... v17\n" 형태로 ESP32에 전송.
    """
    vals = np.clip(norms, -1.0, 1.0) * 1000.0
    vals = vals.astype(int)
    line = "S " + " ".join(f"{v:d}" for v in vals) + "\n"
    # 디버깅용: 필요하면 주석 해제
    # print("TX:", line.strip())
    try:
        ser.write(line.encode("ascii"))
    except serial.SerialException as e:
        print("Serial write error:", e)


def read_from_esp():
    """
    ESP가 보내는 시리얼 라인 읽기:
      - "F ..." : FSR 정규화 값 (float 6개)
      - "C ..." : 다리별 접촉 여부 (0/1, 6개)
    최신값을 last_fsr_norm, last_leg_contact에 저장.
    """
    global last_fsr_norm, last_leg_contact

    # in_waiting 만큼 읽어주면 버퍼가 안 쌓임
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue

            # 디버깅용: 필요하면 출력
            # print("RX:", line)

            if line.startswith("F "):
                parts = line.split()[1:]
                if len(parts) >= NUM_LEGS:
                    try:
                        vals = [float(x) for x in parts[:NUM_LEGS]]
                        last_fsr_norm = np.array(vals, dtype=float)
                    except ValueError:
                        pass  # 숫자 파싱 실패하면 무시

            elif line.startswith("C "):
                parts = line.split()[1:]
                if len(parts) >= NUM_LEGS:
                    try:
                        vals = [int(x) for x in parts[:NUM_LEGS]]
                        last_leg_contact = np.array(vals, dtype=int)
                    except ValueError:
                        pass
    except serial.SerialException as e:
        print("Serial read error:", e)


# ---------- 뷰어 + 시뮬레이션 루프 ----------
with viewer.launch_passive(model, data) as v:
    last_send = time.time()
    debug_print_t = time.time()

    while v.is_running():
        # 한 스텝 진행
        mujoco.mj_step(model, data)

        # ESP → 파이썬: FSR/Contact 정보 읽기
        read_from_esp()

        # 일정 주기로 ESP에 제어 신호 전송
        now = time.time()
        if now - last_send >= 1.0 / SEND_HZ:
            norms = ctrl_to_normalized()
            send_to_esp(norms)
            last_send = now

        # 디버그: 0.5초마다 한 번씩 현재 접촉 상태 출력
        if now - debug_print_t > 0.5:
            print("Contact:", last_leg_contact, "  FSR:", np.round(last_fsr_norm, 3))
            debug_print_t = now

        # 뷰어와 동기화
        v.sync()
        time.sleep(model.opt.timestep)
