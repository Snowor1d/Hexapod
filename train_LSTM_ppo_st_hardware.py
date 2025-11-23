# train_hexapod_global.py
import os
import glob
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sb3_contrib import RecurrentPPO  # 🔹 RecurrentPPO 사용
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from hexapod_hardware import HexapodEnv
#import terminal_observation
import time

# ===============================================================
# 🔧 GLOBAL CONFIGURATION
# ===============================================================
MODE = "teacher"   # "teacher", "student_rl", "collect", "bc_train", "bc_eval"

# Paths
XML_PATH = os.getenv("HEXAPOD_XML", "hexapod_hardware.xml")
LOGDIR   = "./logs_hexapod_hardware_lstm"

# Env Params
N_ENVS = 6
SEED = 42
ACTION_REPEAT = 20
TARGET_SPEED = 0.2
STUDENT_HIST_LEN = 1
CONTACT_THRESHOLD = 1e-4
MAX_STEPS = 500
RENDER = False
WARMUP_STEPS = 50
RANDOM_INIT_POSTURE = False

# PPO Params
NET_ARCH = [64, 64] #96, 96
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
COLLECT_STEPS = 5_000_000
EVAL_STEPS = 3000

# BC Params
BC_HIDDEN = [256, 256]
BC_LR = 1e-3
BC_BATCH = 4096
BC_EPOCHS = 300
BC_VAL_SPLIT = 0.1
FORCE_CPU = False

# DART params
DART_ENABLED = True
DART_SIGMA = 0.2
DART_CLIP = 0.4

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
            random_init_posture=RANDOM_INIT_POSTURE,
        )
        return Monitor(env)
    return _init


def build_ppo(venv, logdir):
    policy_kwargs = dict(
        net_arch=list(NET_ARCH),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        policy_kwargs=policy_kwargs,
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
# 🧠 STUDENT RL (지금은 teacher → student RL 파인튜닝이 아니라, 순수 PPO만)
#    → 논문 방식 teacher-student distillation만 쓸거면 이 함수는 안 써도 됨
# ===============================================================
def train_student_rl():
    run_dir = os.path.join(LOGDIR, "student_rl")
    os.makedirs(run_dir, exist_ok=True)
    if N_ENVS > 1:
        venv = SubprocVecEnv([make_env(i, "student") for i in range(N_ENVS)])
    else:
        venv = DummyVecEnv([make_env(0, "student")])
    venv = VecMonitor(venv, filename=os.path.join(run_dir, "monitor.csv"))

    # 🔴 예전처럼 BC weight을 억지로 복사하지 않고, 그냥 순수 RecurrentPPO 학습
    model = build_ppo(venv, run_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(run_dir, "ppo_student_rl"))
    venv.close()


# ===============================================================
# 🎯 COLLECT (teacher → student pairs + teacher latent Z)
# ===============================================================
def collect_teacher_student_pairs():
    run_dir = os.path.join(LOGDIR, "collect")
    os.makedirs(run_dir, exist_ok=True)

    # 🔹 단일 env
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode="none",
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_clip=10.0,
        seed=SEED,
        obs_mode="teacher_student",
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        random_init_posture=RANDOM_INIT_POSTURE,
    )

    teacher_path = os.path.join(LOGDIR, "teacher", "ppo_teacher.zip")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher policy not found: {teacher_path}")
    teacher = RecurrentPPO.load(teacher_path)

    S_list, A_list, Z_list = [], [], []
    obs, info = env.reset()
    ep_step = 0

    lstm_state = None
    episode_start = True  # 첫 스텝은 새로운 에피소드

    for t in range(COLLECT_STEPS):
        if t % 100000 == 0:
            print(t, " collected")

        # 🔹 student가 실제로 보게 될 관측
        student_obs_t = info["student_obs"].copy()

        # 🔹 teacher 행동 (RecurrentPPO)
        action, lstm_state = teacher.predict(
            obs,
            state=lstm_state,
            episode_start=np.array([episode_start], dtype=bool),
            deterministic=True,
        )

        # 🔹 teacher latent Z 추출
        with torch.no_grad():
            obs_torch, _ = teacher.policy.obs_to_tensor(obs)  # (1, obs_dim)
            features = teacher.policy.extract_features(obs_torch)

            # share_features_extractor=True인 경우 features 그 자체가 policy/critic 공용 피처
            if isinstance(features, tuple):
                pi_features, _ = features
            else:
                pi_features = features

            # 여기서는 "teacher의 actor 입력 피처"를 latent로 사용
            z = pi_features.detach().cpu().numpy()[0].astype(np.float32)

        # 🔹 warmup 이후만 BC 데이터에 사용
        if ep_step >= WARMUP_STEPS:
            S_list.append(student_obs_t)
            A_list.append(action.copy())
            Z_list.append(z)

        # 🔹 DART: noisy action으로 환경 rollout
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
            lstm_state = None
            episode_start = True
        else:
            episode_start = False

    env.close()

    S = np.stack(S_list, axis=0).astype(np.float32)
    A = np.stack(A_list, axis=0).astype(np.float32)
    Z = np.stack(Z_list, axis=0).astype(np.float32)

    out = os.path.join(
        run_dir,
        f"pairs_S{S.shape[-1]}_A{A.shape[-1]}_Z{Z.shape[-1]}_N{S.shape[0]}.npz"
    )
    np.savez_compressed(out, S=S, A=A, Z=Z)
    print(f"[collect] Saved {S.shape[0]} samples -> {out}")


# ===============================================================
# 🤖 BEHAVIOR CLONING (BC) with latent + action distillation
# ===============================================================
class StudentDataset(Dataset):
    def __init__(self, files, validation_split=0.1, mode="train"):
        Ss, As, Zs = [], [], []
        for f in files:
            d = np.load(f)
            if "Z" not in d:
                raise ValueError(f"{f} has no 'Z' array. Re-run collect after updating code.")
            Ss.append(d["S"])
            As.append(d["A"])
            Zs.append(d["Z"])

        self.S = np.concatenate(Ss, 0).astype(np.float32)
        self.A = np.concatenate(As, 0).astype(np.float32)
        self.Z = np.concatenate(Zs, 0).astype(np.float32)

        num_samples = self.S.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        split_idx = int(num_samples * (1 - validation_split))

        if mode == "train":
            self.indices = indices[:split_idx]
        elif mode == "val":
            self.indices = indices[split_idx:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.S[idx], self.A[idx], self.Z[idx]


class StudentNet(nn.Module):
    """
    Student: 제한된 관측 S -> latent Z_hat, action A_hat
    논문처럼 latent + action을 동시에 distill.
    """
    def __init__(self, in_dim, latent_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.latent_head = nn.Linear(last, latent_dim)
        self.action_head = nn.Linear(last, act_dim)

    def forward(self, x):
        h = self.backbone(x)
        z_hat = self.latent_head(h)
        a_hat = torch.tanh(self.action_head(h))
        return z_hat, a_hat


def bc_train():
    collect_dir = os.path.join(LOGDIR, "collect")
    files = sorted(glob.glob(os.path.join(collect_dir, "*.npz")))
    if not files:
        raise FileNotFoundError("No collected .npz found in ./logs_hexapod_hardware_lstm/collect")

    train_ds = StudentDataset(files, validation_split=BC_VAL_SPLIT, mode="train")
    val_ds = StudentDataset(files, validation_split=BC_VAL_SPLIT, mode="val")

    Sdim = train_ds.S.shape[1]
    Adim = train_ds.A.shape[1]
    Zdim = train_ds.Z.shape[1]
    print(f"[BC Train] Train N={len(train_ds)}, Val N={len(val_ds)}, Sdim={Sdim}, Adim={Adim}, Zdim={Zdim}")

    train_loader = DataLoader(train_ds, batch_size=BC_BATCH, shuffle=True,
                              drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BC_BATCH, shuffle=False,
                            drop_last=False, num_workers=4)

    dev = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    net = StudentNet(Sdim, Zdim, Adim, BC_HIDDEN).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=BC_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    outdir = os.path.join(LOGDIR, "student_bc")
    os.makedirs(outdir, exist_ok=True)
    best_val_loss = float("inf")
    ckpt = os.path.join(outdir, "student_bc.pt")

    for ep in range(1, BC_EPOCHS + 1):
        # --- Train ---
        loss_sum = 0.0
        net.train()
        for S, A, Z in train_loader:
            S, A, Z = S.to(dev), A.to(dev), Z.to(dev)
            z_hat, a_hat = net(S)

            loss_z = ((z_hat - Z) ** 2).mean()
            loss_a = ((a_hat - A) ** 2).mean()
            loss = loss_z + loss_a  # 필요하면 가중치 넣어도 됨

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()

        avg_train_loss = loss_sum / len(train_loader)

        # --- Validation ---
        val_loss_sum = 0.0
        net.eval()
        with torch.no_grad():
            for S_val, A_val, Z_val in val_loader:
                S_val, A_val, Z_val = S_val.to(dev), A_val.to(dev), Z_val.to(dev)
                z_hat_val, a_hat_val = net(S_val)
                loss_z_val = ((z_hat_val - Z_val) ** 2).mean()
                loss_a_val = ((a_hat_val - A_val) ** 2).mean()
                val_loss_sum += (loss_z_val + loss_a_val).item()

        avg_val_loss = val_loss_sum / len(val_loader)
        sched.step()

        print(f"[BC] epoch {ep}/{BC_EPOCHS} Train={avg_train_loss:.6f}  Val={avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "model": net.state_dict(),
                    "Sdim": Sdim,
                    "Adim": Adim,
                    "Zdim": Zdim,
                    "hidden": BC_HIDDEN,
                },
                ckpt
            )
            print(f"  ↳ saved best ckpt (Val={best_val_loss:.6f}) -> {ckpt}")


def bc_eval():
    ckpt = os.path.join(LOGDIR, "student_bc", "student_bc.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No BC checkpoint at {ckpt}")
    d = torch.load(ckpt, map_location="cpu")
    net = StudentNet(d["Sdim"], d["Zdim"], d["Adim"], d["hidden"])
    net.load_state_dict(d["model"])
    dev = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    net.to(dev).eval()

    # 🔹 student obs로 동작하는 정책으로 평가
    env = make_env(0, "student")()

    obs, _ = env.reset()
    ret, steps = 0.0, 0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            S = torch.from_numpy(obs).unsqueeze(0).to(dev)
            _, a = net(S)          # latent는 버리고 action만 사용
            a = a.cpu().numpy()[0]
            obs, r, done, trunc, info = env.step(a)
            ret += r
            steps += 1
            if RENDER:
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
