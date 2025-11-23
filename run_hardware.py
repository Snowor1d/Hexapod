# run_hexapod.py
import os
import numpy as np
import torch
import time
import serial   # 🔹 추가: pyserial

# ===============================================================
# ⚙️ GLOBAL CONFIGURATION
# ===============================================================
XML_PATH   = "hexapod_hardware.xml"
MODEL_PATH = "./logs_hexapod_hardware_no_contact/student_bc/student_bc.pt"
#MODEL_PATH = "./logs_hexapod_hardware/student_bc/student_bc_epoch_120.pt"
#MODEL_PATH = "./logs_hexapod_hardware/teacher/ppo_teacher.zip"

# Environment settings
OBS_MODE           = "student"   # "teacher" | "student" | "teacher_student"
RENDER             = True
EPISODES           = 5
DETERMINISTIC      = False       # PPO 전용 (deterministic=True)
ACTION_REPEAT      = 20
TARGET_SPEED       = 0.25
STUDENT_HIST_LEN   = 1
CONTACT_THRESHOLD  = 1e-4
MAX_STEPS          = 500
STUDENT_INCLUDE_IMU = False

# 🔹 ESP32 시리얼 설정 (포트/보드에 맞게 수정!)
SERIAL_PORT = "/dev/tty.usbserial-0001"  # 예: 리눅스
# SERIAL_PORT = "COM3"        # 예: 윈도우
BAUD_RATE   = 115200

# ===============================================================
# 🧩 ENV IMPORT
# ===============================================================
from hexapod_hardware import HexapodEnv

# ===============================================================
# 📦 MODEL LOADERS
# ===============================================================
def load_ppo(model_path, venv):
    from stable_baselines3 import PPO
    model = PPO.load(model_path, env=venv)
    return model

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [torch.nn.Linear(last,h), torch.nn.ReLU()]
            last = h
        layers += [torch.nn.Linear(last,out_dim), torch.nn.Tanh()]
        self.net = torch.nn.Sequential(*layers)
    def forward(self,x): 
        return self.net(x)

def load_bc(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu")
    net  = MLP(ckpt["Sdim"], ckpt["Adim"], tuple(ckpt.get("hidden",(256,256))))
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    return net, ckpt["Sdim"], ckpt["Adim"]

# ===============================================================
# 🔌 SERIAL HELPERS (Python → ESP, ESP → Python)
# ===============================================================
def open_serial(port, baud):
    ser = serial.Serial(port, baudrate=baud, timeout=0.01)
    # 보드가 리셋될 시간을 조금 준다
    time.sleep(2.0)
    # 초기 버퍼 비우기
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print(f"[INFO] Opened serial: {port} @ {baud}")
    return ser

def send_action_to_esp(ser, act):
    """
    act: numpy array, shape (18,), range [-1, 1]
    ESP 쪽 포맷: 'S v0 v1 ... v17\\n', vi in [-1000,1000]
    """
    if ser is None:
        return

    act = np.asarray(act, dtype=np.float32)
    act = np.clip(act, -1.0, 1.0)
    vals = np.round(act * 1000.0).astype(int)

    if vals.shape[0] != 18:
        # 혹시 action 차원이 바뀌었으면 경고만 찍고 전송 생략
        print(f"[WARN] Action dim != 18 (got {vals.shape[0]}), skip send.")
        return

    line = "S " + " ".join(str(v) for v in vals) + "\n"
    ser.write(line.encode("ascii"))
    # 필요하다면 flush
    # ser.flush()

def read_esp_lines(ser):
    """
    ESP에서 오는 F/C 라인(또는 디버그 메시지)을 읽어서 출력.
    꼭 필요하진 않지만, 버퍼가 꽉 차는 것 방지 + 상태 확인용.
    """
    if ser is None:
        return

    try:
        # 한 번에 여러 줄 읽을 수도 있음
        while ser.in_waiting:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line:
                continue
            # 여기서 패턴에 따라 파싱할 수 있음.
            # 예: "F ..." / "C ..." / 기타 로그
            print(f"[ESP] {line}")
    except Exception as e:
        print(f"[WARN] Serial read error: {e}")

# ===============================================================
# 🚀 POLICY RUNNER
# ===============================================================
def run_policy(policy_kind, policy, env, episodes=5, deterministic=True, device="cpu", ser=None):
    ep_stats = []
    for ep in range(1, episodes+1):
        obs, info = env.reset()

        # BC → student_obs 사용
        use_student_obs = (policy_kind == "bc")
        if use_student_obs and env.obs_mode != "student":
            obs_infer = info.get("student_obs", obs)
        else:
            obs_infer = obs

        ret = 0.0
        steps = 0
        xs = []

        while True:
            # ====== 1) 정책에서 action 생성 ======
            if policy_kind == "ppo":
                act, _ = policy.predict(obs, deterministic=deterministic)
            else:
                with torch.no_grad():
                    o = torch.from_numpy(obs_infer).float().unsqueeze(0).to(device)
                    act = policy(o).cpu().numpy()[0]

            # ====== 2) action을 ESP로 전송 ======
            send_action_to_esp(ser, act)

            # (옵션) ESP에서 오는 센서/로그 읽기
            read_esp_lines(ser)

            # ====== 3) 시뮬레이션/환경에도 같은 action 적용 ======
            obs, r, done, trunc, info = env.step(act)
            ret  += float(r)
            steps += 1
            xs.append(float(info.get("xd", 0.0)))

            # 하드웨어가 너무 급하게 안 움직이도록 약간 텀
            time.sleep(0.03)

            # BC만 student obs 사용
            if policy_kind == "bc":
                if env.obs_mode == "student":
                    obs_infer = obs
                else:
                    obs_infer = info.get("student_obs", obs)

            if done or trunc:
                mean_xd = float(np.mean(xs)) if xs else 0.0
                ep_stats.append((ret, steps, mean_xd))
                print(f"[Episode {ep}] Return={ret:.2f}  Steps={steps}  mean_vx={mean_xd:.3f} m/s")
                break
    return ep_stats

# ===============================================================
# 🧠 MAIN
# ===============================================================
def main():
    render_mode = "human" if RENDER else "none"

    # 🔹 시리얼 포트 오픈 (ESP32 연결)
    try:
        ser = open_serial(SERIAL_PORT, BAUD_RATE)
    except Exception as e:
        print(f"[WARN] Could not open serial port {SERIAL_PORT}: {e}")
        ser = None

    # Env 생성 (시뮬레이션 / 하드웨어용 obs 계산)
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode=render_mode,
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_mode=OBS_MODE,
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=MAX_STEPS,
        random_init_posture=False,
        joint_init_std_deg=5.0,
        yaw_init_std_deg=10.0,
        student_include_imu=STUDENT_INCLUDE_IMU,
    )

    # 모델 확장자 확인
    ext = os.path.splitext(MODEL_PATH)[1].lower()

    if ext == ".zip":     # PPO
        from stable_baselines3 import PPO
        policy = load_ppo(MODEL_PATH, env)
        print(f"[INFO] Loaded PPO model: {MODEL_PATH}")
        run_policy("ppo", policy, env,
                   episodes=EPISODES,
                   deterministic=DETERMINISTIC,
                   ser=ser)

    elif ext == ".pt":    # BC
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, Sdim, Adim = load_bc(MODEL_PATH, device)
        print(f"[INFO] Loaded BC model: {MODEL_PATH}")
        ob, info = env.reset()
        if OBS_MODE == "student":
            cur_dim = ob.shape[0]
        else:
            cur_dim = info.get("student_obs", ob).shape[0]
        if cur_dim != Sdim:
            print(f"[WARN] Env obs_dim ({cur_dim}) != model input ({Sdim}) → hist_len/obs_mode 확인 필요")
        run_policy("bc", net, env,
                   episodes=EPISODES,
                   device=device,
                   ser=ser)

    else:
        raise ValueError(f"Unsupported model type: {MODEL_PATH}")

    env.close()

    if ser is not None:
        ser.close()
        print("[INFO] Serial closed.")

if __name__ == "__main__":
    main()
