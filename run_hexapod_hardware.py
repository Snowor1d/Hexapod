# run_hexapod.py
import os
import numpy as np
import torch
import time
# ===============================================================
# ⚙️ GLOBAL CONFIGURATION
# ===============================================================
XML_PATH = "hexapod_hardware.xml"
#MODEL_PATH = "./logs_hexapod_hardware/student_bc/student_bc_epoch_120.pt"   # PPO: .zip / BC: .pt
MODEL_PATH = "./logs_hexapod_hardware_no_contact/student_bc/student_bc.pt" 
#MODEL_PATH = "./logs_hexapod_hardware_no_contact/teacher/ppo_teacher.zip"

# Environment settings
OBS_MODE = "student"            # "teacher" | "student" | "teacher_student"
RENDER = True
EPISODES = 5
DETERMINISTIC = False            # PPO 전용 (deterministic=True)
ACTION_REPEAT = 20
TARGET_SPEED = 0.25
STUDENT_HIST_LEN = 1
CONTACT_THRESHOLD = 1e-4
MAX_STEPS = 500
STUDENT_INCLUDE_IMU = False

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
    def forward(self,x): return self.net(x)

def load_bc(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu")
    net  = MLP(ckpt["Sdim"], ckpt["Adim"], tuple(ckpt.get("hidden",(256,256))))
    net.load_state_dict(ckpt["model"])
    net.to(device).eval()
    return net, ckpt["Sdim"], ckpt["Adim"]

# ===============================================================
# 🚀 POLICY RUNNER
# ===============================================================
def run_policy(policy_kind, policy, env, episodes=5, deterministic=True, device="cpu"):
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
            if policy_kind == "ppo":
                act, _ = policy.predict(obs, deterministic=deterministic)
            else:
                with torch.no_grad():
                    o = torch.from_numpy(obs_infer).float().unsqueeze(0).to(device)
                    act = policy(o).cpu().numpy()[0]

            obs, r, done, trunc, info = env.step(act)
            ret  += float(r)
            steps += 1
            xs.append(float(info.get("xd", 0.0)))
            time.sleep(0.01)
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

    # Env 생성
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode=render_mode,
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_mode=OBS_MODE,
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=MAX_STEPS,
        random_init_posture = False,
        joint_init_std_deg = 5.0,
        yaw_init_std_deg = 10.0,
        student_include_imu = STUDENT_INCLUDE_IMU
        
    )

    # 모델 확장자 확인
    ext = os.path.splitext(MODEL_PATH)[1].lower()

    if ext == ".zip":     # PPO
        from stable_baselines3 import PPO
        policy = load_ppo(MODEL_PATH, env)
        print(f"[INFO] Loaded PPO model: {MODEL_PATH}")
        run_policy("ppo", policy, env, episodes=EPISODES, deterministic=DETERMINISTIC)

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
        run_policy("bc", net, env, episodes=EPISODES, device=device)

    else:
        raise ValueError(f"Unsupported model type: {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()