# run_trained_policy.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from hexapod_real import HexapodEnv

# =====================================================
# 🔧 GLOBAL CONFIG
# =====================================================
MODE = "student_rl"          # "teacher" | "student_bc" | "student_rl"
RENDER = True
MAX_STEPS = 50_000
SLEEP_SEC = 0.01          # 렌더 속도 조절(0이면 최대속도)

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE, "hexapod.xml")
MODEL_TEACHER = os.path.join(BASE, "logs_hexapod", "teacher", "ppo_teacher.zip")
MODEL_STUDENT_RL = os.path.join(BASE, "logs_hexapod", "student_rl", "ppo_student_rl.zip")
MODEL_STUDENT_BC = os.path.join(BASE, "logs_hexapod", "student_bc", "student_bc.pt")

# Env params (학습 때와 동일하게 맞추기)
ACTION_REPEAT = 20
TARGET_SPEED = 0.35
STUDENT_HIST_LEN = 10
CONTACT_THRESHOLD = 1e-4
EPISODE_MAX_STEPS = 500
SEED = 0

# =====================================================
# 🧠 Student-BC MLP (run-time용)
# =====================================================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.Tanh()]  # [-1,1]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# =====================================================
# 🚀 Runner
# =====================================================
def run_teacher():
    if not os.path.exists(MODEL_TEACHER):
        raise FileNotFoundError(f"Teacher model not found: {MODEL_TEACHER}")
    print(f"[Teacher] Loading: {MODEL_TEACHER}")

    model = PPO.load(MODEL_TEACHER)
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode="human" if RENDER else "none",
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_mode="teacher",
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=EPISODE_MAX_STEPS,
        seed=SEED
    )
    obs, _ = env.reset()
    print("Simulation started (Teacher). Ctrl+C to exit.")
    try:
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if SLEEP_SEC: time.sleep(SLEEP_SEC)
            if terminated or truncated:
                print("Episode ended. Resetting...")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    env.close()

def run_student_rl():
    if not os.path.exists(MODEL_STUDENT_RL):
        raise FileNotFoundError(f"Student-RL model not found: {MODEL_STUDENT_RL}")
    print(f"[Student-RL] Loading: {MODEL_STUDENT_RL}")

    model = PPO.load(MODEL_STUDENT_RL)
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode="human" if RENDER else "none",
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_mode="student",
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=EPISODE_MAX_STEPS,
        seed=SEED
    )
    obs, _ = env.reset()
    print("Simulation started (Student-RL). Ctrl+C to exit.")
    try:
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if SLEEP_SEC: time.sleep(SLEEP_SEC)
            if terminated or truncated:
                print("Episode ended. Resetting...")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    env.close()

def run_student_bc():
    if not os.path.exists(MODEL_STUDENT_BC):
        raise FileNotFoundError(f"Student-BC checkpoint not found: {MODEL_STUDENT_BC}")
    print(f"[Student-BC] Loading: {MODEL_STUDENT_BC}")

    # 로드(학습 스크립트에서 저장한 포맷 가정: {"model", "Sdim", "Adim", "hidden"})
    ckpt = torch.load(MODEL_STUDENT_BC, map_location="cpu")
    Sdim, Adim = ckpt["Sdim"], ckpt["Adim"]
    hidden = tuple(ckpt.get("hidden", (256, 256)))
    net = MLP(Sdim, Adim, hidden)
    net.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode="human" if RENDER else "none",
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_mode="student",  # student 관측만 사용
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=EPISODE_MAX_STEPS,
        seed=SEED
    )
    obs, _ = env.reset()
    print("Simulation started (Student-BC). Ctrl+C to exit.")
    try:
        with torch.no_grad():
            for step in range(MAX_STEPS):
                S = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = net(S).cpu().numpy()[0]   # [-1,1]
                obs, reward, terminated, truncated, info = env.step(action)
                if SLEEP_SEC: time.sleep(SLEEP_SEC)
                if terminated or truncated:
                    print("Episode ended. Resetting...")
                    obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    env.close()

def main():
    print(f"XML: {XML_PATH}")
    if MODE == "teacher":
        run_teacher()
    elif MODE == "student_rl":
        run_student_rl()
    elif MODE == "student_bc":
        run_student_bc()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()
