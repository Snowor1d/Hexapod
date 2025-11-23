# train_hexapod_global.py
import os
import glob
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from hexapod_hardware import HexapodEnv
#import terminal_observation
import time

# ===============================================================
# 🔧 GLOBAL CONFIGURATION
# ===============================================================
MODE = "bc_train"   # "teacher", "student_rl", "collect", "bc_train", "bc_eval"

# Paths
# ⬇️ 환경변수로 XML 교체 가능: export HEXAPOD_XML=hexapod.xml
XML_PATH = os.getenv("HEXAPOD_XML", "hexapod_hardware.xml")
LOGDIR   = "./logs_hexapod_hardware"

# Env Params (XML과 합맞춤: imu_site/foot_site1~6, keyframe 'home' 사용 가정)
N_ENVS = 6
SEED = 42
ACTION_REPEAT = 20
TARGET_SPEED = 0.1
STUDENT_HIST_LEN = 1
CONTACT_THRESHOLD = 1e-4
MAX_STEPS = 500
RENDER = False  # 필요 시 True로
WARMUP_STEPS = 50
RANDOM_INIT_POSTURE = False

# PPO Params
NET_ARCH = [96, 96]
LR = 3e-4
N_STEPS = 1024
BATCH_SIZE = 1024
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.001
VF_COEF = 0.5
TOTAL_TIMESTEPS = 6_000_000

# Collection & Evaluation
COLLECT_STEPS = 3_000_000
EVAL_STEPS = 3000

# BC Params
BC_HIDDEN = [256, 256]
BC_LR = 1e-3
BC_BATCH = 4096
BC_EPOCHS = 20
BC_VAL_SPLIT = 0.1
FORCE_CPU = False

#Dart params
DART_ENABLED = True
DART_SIGMA = 0.2
DART_CLIP = 0.4

STUDENT_INCLUDE_IMU = True

torch.set_num_threads(4)
os.environ.setdefault("OMP_NUM_THREADS", "4")


# ===============================================================
# 🧩 ENV FACTORY
# ===============================================================
def make_env(rank, obs_mode):
    """
    obs_mode: "teacher" | "student" | "teacher_student"
    """
    def _init():
        env = HexapodEnv(
            xml_path=XML_PATH,
            render_mode="none",
            action_repeat=ACTION_REPEAT,
            target_speed=TARGET_SPEED,
            obs_clip=10.0,
            seed=SEED + rank,
            obs_mode=obs_mode,
            student_hist_len=STUDENT_HIST_LEN,
            contact_threshold=CONTACT_THRESHOLD,
            max_steps=MAX_STEPS,
            warmup_steps=WARMUP_STEPS,
            random_init_posture = RANDOM_INIT_POSTURE,
            student_include_imu = STUDENT_INCLUDE_IMU
        )
        return Monitor(env)
    return _init


def build_ppo(venv, logdir):
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=list(NET_ARCH)),
        learning_rate=LR,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        verbose=1,
    )
    logger = configure(logdir, ["tensorboard", "stdout"])
    model.set_logger(logger)
    return model


# ===============================================================
# 🧠 TRAIN TEACHER
# ===============================================================
def train_teacher():
    run_dir = os.path.join(LOGDIR, "teacher")
    os.makedirs(run_dir, exist_ok=True)
    if N_ENVS > 1:
        venv = SubprocVecEnv([make_env(i, "teacher") for i in range(N_ENVS)])
    else:
        venv = DummyVecEnv([make_env(0, "teacher")])
    venv = VecMonitor(venv, filename=os.path.join(run_dir, "monitor.csv"))

    model = build_ppo(venv, run_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(run_dir, "ppo_teacher"))
    venv.close()


# ===============================================================
# 🧠 STUDENT RL
# ===============================================================
def train_student_rl():
    run_dir = os.path.join(LOGDIR, "student_rl")
    os.makedirs(run_dir, exist_ok=True)
    if N_ENVS > 1:
        venv = SubprocVecEnv([make_env(i, "student") for i in range(N_ENVS)])
    else:
        venv = DummyVecEnv([make_env(0, "student")])
    venv = VecMonitor(venv, filename=os.path.join(run_dir, "monitor.csv"))

    model = build_ppo(venv, run_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(run_dir, "ppo_student_rl"))
    venv.close()

# ===============================================================
# 🎯 COLLECT (teacher → student pairs) [수정됨: 병렬 처리]
# ===============================================================
def collect_teacher_student_pairs():
    run_dir = os.path.join(LOGDIR, "collect")
    os.makedirs(run_dir, exist_ok=True)

    # 🔹 단일 환경 사용 (vecenv 안 씀)
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode="none",
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_clip=10.0,
        seed=SEED,
        obs_mode="teacher_student",      # teacher obs + info["student_obs"]
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        student_include_imu = STUDENT_INCLUDE_IMU
        # random_init_posture=True,      # 랜덤 초기자세 쓰고 싶으면 켜기
    )

    teacher_path = os.path.join(LOGDIR, "teacher", "ppo_teacher.zip")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher policy not found: {teacher_path}")
    from stable_baselines3 import PPO
    teacher = PPO.load(teacher_path)

    S_list, A_list = [], []
    obs, info = env.reset()
    ep_step = 0

    for t in range(COLLECT_STEPS):
        # 🔹 student가 실제로 보게 될 관측
        if(t%100000 ==0):
            print(t, " collected")
        student_obs_t = info["student_obs"].copy()

        # 🔹 teacher는 teacher_obs(=obs)를 보고 행동
        action, _ = teacher.predict(obs, deterministic=False)

        # 🔹 warmup 이후 구간만 BC 데이터에 넣기
        if ep_step >= WARMUP_STEPS:
            S_list.append(student_obs_t)
            A_list.append(action.copy())

        # 🔹 DART: noisy action으로 환경을 굴리고,
        #     label은 clean teacher action으로 유지
        if DART_ENABLED:
            noise = np.random.normal(0.0, DART_SIGMA, size=action.shape).astype(np.float32)
            noise = np.clip(noise, -DART_CLIP, DART_CLIP)
            step_action = np.clip(action + noise, -1.0, 1.0)
        else:
            step_action = action

        obs, r, done, trunc, info = env.step(step_action)
        ep_step += 1

        if done or trunc:
            obs, info = env.reset()
            ep_step = 0

    env.close()

    S = np.stack(S_list, axis=0).astype(np.float32)
    A = np.stack(A_list, axis=0).astype(np.float32)
    out = os.path.join(
        run_dir,
        f"pairs_S{S.shape[-1]}_A{A.shape[-1]}_N{S.shape[0]}.npz"
    )
    np.savez_compressed(out, S=S, A=A)
    print(f"[collect] Saved {S.shape[0]} samples -> {out}")

# ===============================================================
# 🤖 BEHAVIOR CLONING (BC)
# ===============================================================
class StudentDataset(Dataset):
    def __init__(self, files, validation_split=0.1, mode="train"):
        Ss, As = [], []
        for f in files:
            d = np.load(f)
            Ss.append(d["S"]); As.append(d["A"])
        
        self.S = np.concatenate(Ss, 0).astype(np.float32)
        self.A = np.concatenate(As, 0).astype(np.float32)
        
        # 1. 훈련/검증 인덱스 분할
        num_samples = self.S.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices) # 데이터 섞기
        
        split_idx = int(num_samples * (1 - validation_split))
        
        if mode == "train":
            self.indices = indices[:split_idx]
        elif mode == "val":
            self.indices = indices[split_idx:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self): 
        return len(self.indices) # 분할된 인덱스 개수 반환
    
    def __getitem__(self, i): 
        idx = self.indices[i] # 분할된 인덱스 사용
        return self.S[idx], self.A[idx]


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def bc_train():
    collect_dir = os.path.join(LOGDIR, "collect")
    files = sorted(glob.glob(os.path.join(collect_dir, "*.npz")))
    if not files:
        raise FileNotFoundError("No collected .npz found in ./logs_hexapod_hardware/collect")
    
    # 2. 훈련/검증 데이터셋 생성
    train_ds = StudentDataset(files, validation_split=BC_VAL_SPLIT, mode="train")
    val_ds = StudentDataset(files, validation_split=BC_VAL_SPLIT, mode="val")
    
    Sdim, Adim = train_ds.S.shape[1], train_ds.A.shape[1]
    print(f"[BC Train] Train N={len(train_ds)}, Val N={len(val_ds)}, Sdim={Sdim}, Adim={Adim}")

    # 3. 데이터 로더 생성
    train_loader = DataLoader(train_ds, batch_size=BC_BATCH, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BC_BATCH, shuffle=False, drop_last=False, num_workers=4)
    
    dev = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    net = MLP(Sdim, Adim, BC_HIDDEN).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=BC_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    outdir = os.path.join(LOGDIR, "student_bc")
    os.makedirs(outdir, exist_ok=True)
    best_val_loss = float("inf")
    ckpt = os.path.join(outdir, "student_bc.pt")

    # 🔹 여기서부터: loss 히스토리 저장용 리스트
    train_history = []
    val_history = []

    for ep in range(1, BC_EPOCHS + 1):
        # --- 훈련 ---
        loss_sum = 0.0
        net.train()
        for S, A in train_loader:
            S, A = S.to(dev), A.to(dev)
            pred = net(S)
            loss = ((pred - A) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        
        avg_train_loss = loss_sum / len(train_loader)
        
        # --- 검증 ---
        val_loss_sum = 0.0
        net.eval()
        with torch.no_grad():
            for S_val, A_val in val_loader:
                S_val, A_val = S_val.to(dev), A_val.to(dev)
                pred_val = net(S_val)
                val_loss = ((pred_val - A_val) ** 2).mean()
                val_loss_sum += val_loss.item()
        
        avg_val_loss = val_loss_sum / len(val_loader)
        sched.step()
        
        # 🔹 히스토리에 저장
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)

        print(f"[BC] epoch {ep}/{BC_EPOCHS} "
              f"Train_MSE={avg_train_loss:.6f}  Val_MSE={avg_val_loss:.6f}")
        
        # 검증 손실 기준으로 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {"model": net.state_dict(), "Sdim": Sdim, "Adim": Adim, "hidden": BC_HIDDEN},
                ckpt
            )
            print(f"  ↳ saved best ckpt (Val_MSE={best_val_loss:.6f}) -> {ckpt}")

    # 🔹 학습 끝난 후 loss 기록 저장 (npz)
    hist_path = os.path.join(outdir, "bc_loss_history.npz")
    np.savez_compressed(
        hist_path,
        train=np.array(train_history, dtype=np.float32),
        val=np.array(val_history, dtype=np.float32),
    )
    print(f"[BC] Saved loss history -> {hist_path}")


def bc_eval():
    ckpt = os.path.join(LOGDIR, "student_bc", "student_bc.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("No BC checkpoint at logs_hexapod_hardware/student_bc/student_bc.pt")
    d = torch.load(ckpt, map_location="cpu")
    net = MLP(d["Sdim"], d["Adim"], d["hidden"])
    net.load_state_dict(d["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    net.to(dev).eval()

    # 6. make_env 팩토리 사용 (일관성)
    # RENDER=True이면 rank=0이므로 human 모드로 자동 설정됨
    env = make_env(0, "student", use_monitor=False)()

    obs, _ = env.reset()
    ret, steps = 0.0, 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            S = torch.from_numpy(obs).unsqueeze(0).to(dev)
            a = net(S).cpu().numpy()[0]
            obs, r, done, trunc, info = env.step(a)
            ret += r
            steps += 1
            if RENDER: # 렌더링 시 약간의 딜레이
                time.sleep(0.01)
            if done or trunc:
                print(f"[BC Eval] Return={ret:.2f} Steps={steps}")
                ret, steps = 0.0, 0
                obs, _ = env.reset()
    env.close()

# ===============================================================
# 🚀 MAIN
# ===============================================================
if __name__ == "__main__":
    # MacOS에서 SubprocVecEnv 사용 시 필수
    mp.set_start_method("spawn", force=True)
    os.makedirs(LOGDIR, exist_ok=True)

    if MODE == "teacher":
        train_teacher()
    elif MODE == "student_rl":
        train_student_rl()
    elif MODE == "collect":
        collect_teacher_student_pairs()
    elif MODE == "bc_train":
        bc_train()
    elif MODE == "bc_eval":
        bc_eval()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
