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
from hexapod_real import HexapodEnv

# ===============================================================
# 🔧 GLOBAL CONFIGURATION
# ===============================================================
MODE = "student_rl"            # "teacher", "student_rl", "collect", "bc_train", "bc_eval"

# Paths
LOGDIR = "./logs_hexapod"
XML_PATH = "hexapod_uneven.xml"

# Env Params
N_ENVS = 4
SEED = 42
ACTION_REPEAT = 20
TARGET_SPEED = 0.35
STUDENT_HIST_LEN = 10
CONTACT_THRESHOLD = 1e-4
MAX_STEPS = 500

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
TOTAL_TIMESTEPS = 2_000_000

# Collection & Evaluation
COLLECT_STEPS = 300_000
EVAL_STEPS = 3000
RENDER = True

# BC Params
BC_HIDDEN = [256, 256]
BC_LR = 1e-3
BC_BATCH = 4096
BC_EPOCHS = 50
FORCE_CPU = False

torch.set_num_threads(4)
os.environ.setdefault("OMP_NUM_THREADS", "4")

# ===============================================================
# 🧩 ENV FACTORY
# ===============================================================
def make_env(rank, obs_mode):
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
    venv = SubprocVecEnv([make_env(i, "teacher") for i in range(N_ENVS)]) if N_ENVS > 1 \
        else DummyVecEnv([make_env(0, "teacher")])
    venv = VecMonitor(venv, filename=os.path.join(run_dir, "monitor.csv"))
    model = build_ppo(venv, run_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(run_dir, "ppo_teacher"))
    venv.close()

# ===============================================================
# 🧠 STUDENT RL (optional)
# ===============================================================
def train_student_rl():
    run_dir = os.path.join(LOGDIR, "student_rl")
    os.makedirs(run_dir, exist_ok=True)
    venv = SubprocVecEnv([make_env(i, "student") for i in range(N_ENVS)]) if N_ENVS > 1 \
        else DummyVecEnv([make_env(0, "student")])
    venv = VecMonitor(venv, filename=os.path.join(run_dir, "monitor.csv"))
    model = build_ppo(venv, run_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(os.path.join(run_dir, "ppo_student_rl"))
    venv.close()

# ===============================================================
# 🎯 COLLECT (teacher → student pairs)
# ===============================================================
def collect_teacher_student_pairs():
    run_dir = os.path.join(LOGDIR, "collect")
    os.makedirs(run_dir, exist_ok=True)
    env = HexapodEnv(
        XML_PATH, render_mode="none",
        action_repeat=ACTION_REPEAT,
        target_speed=TARGET_SPEED,
        obs_mode="teacher_student",
        student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD,
        max_steps=MAX_STEPS,
    )
    teacher_path = os.path.join(LOGDIR, "teacher", "ppo_teacher.zip")
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher policy not found: {teacher_path}")
    teacher = PPO.load(teacher_path)

    obs, info = env.reset()
    S_list, A_list, R_list = [], [], []
    for t in range(COLLECT_STEPS):
        action, _ = teacher.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        S_list.append(info["student_obs"].copy())
        A_list.append(action.copy())
        R_list.append(r)
        if done or trunc:
            obs, info = env.reset()
    env.close()

    S = np.stack(S_list, axis=0)
    A = np.stack(A_list, axis=0)
    np.savez_compressed(
        os.path.join(run_dir, f"pairs_S{S.shape[-1]}_A{A.shape[-1]}_N{S.shape[0]}.npz"),
        S=S, A=A
    )
    print(f"[collect] Saved {S.shape[0]} samples.")

# ===============================================================
# 🤖 BEHAVIOR CLONING (BC)
# ===============================================================
class StudentDataset(Dataset):
    def __init__(self, files):
        Ss, As = [], []
        for f in files:
            d = np.load(f)
            Ss.append(d["S"]); As.append(d["A"])
        self.S = np.concatenate(Ss, 0).astype(np.float32)
        self.A = np.concatenate(As, 0).astype(np.float32)
    def __len__(self): return self.S.shape[0]
    def __getitem__(self, i): return self.S[i], self.A[i]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256,256)):
        super().__init__()
        layers, last = [], in_dim
        for h in hidden:
            layers += [nn.Linear(last,h), nn.ReLU()]; last = h
        layers += [nn.Linear(last,out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def bc_train():
    collect_dir = os.path.join(LOGDIR, "collect")
    files = sorted(glob.glob(os.path.join(collect_dir, "*.npz")))
    if not files: raise FileNotFoundError("No collected .npz found.")
    ds = StudentDataset(files)
    N, Sdim, Adim = len(ds), ds.S.shape[1], ds.A.shape[1]
    print(f"[BC Train] N={N}, Sdim={Sdim}, Adim={Adim}")

    dl = DataLoader(ds, batch_size=BC_BATCH, shuffle=True, drop_last=True)
    dev = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    net = MLP(Sdim, Adim, BC_HIDDEN).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=BC_LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=BC_EPOCHS)

    outdir = os.path.join(LOGDIR, "student_bc")
    os.makedirs(outdir, exist_ok=True)
    best = float("inf"); ckpt = os.path.join(outdir,"student_bc.pt")

    for ep in range(1, BC_EPOCHS+1):
        loss_sum=0
        for S,A in dl:
            S,A=S.to(dev),A.to(dev)
            pred=net(S); loss=((pred-A)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum+=loss.item()
        sched.step()
        avg=loss_sum/len(dl)
        print(f"[BC] epoch {ep}/{BC_EPOCHS} MSE={avg:.6f}")
        if avg<best:
            best=avg
            torch.save({"model":net.state_dict(),"Sdim":Sdim,"Adim":Adim,"hidden":BC_HIDDEN}, ckpt)
            print(f"  ↳ saved best ckpt ({best:.6f})")

def bc_eval():
    ckpt = os.path.join(LOGDIR,"student_bc","student_bc.pt")
    if not os.path.exists(ckpt): raise FileNotFoundError("No BC checkpoint.")
    d=torch.load(ckpt,map_location="cpu")
    net=MLP(d["Sdim"],d["Adim"],d["hidden"])
    net.load_state_dict(d["model"])
    dev=torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    net.to(dev).eval()

    env = HexapodEnv(
        XML_PATH, render_mode="human" if RENDER else "none",
        action_repeat=ACTION_REPEAT, target_speed=TARGET_SPEED,
        obs_mode="student", student_hist_len=STUDENT_HIST_LEN,
        contact_threshold=CONTACT_THRESHOLD, max_steps=MAX_STEPS,
    )
    obs,_=env.reset(); ret,steps=0,0
    with torch.no_grad():
        for _ in range(EVAL_STEPS):
            S=torch.from_numpy(obs).unsqueeze(0).to(dev)
            a=net(S).cpu().numpy()[0]
            obs,r,done,trunc,info=env.step(a)
            ret+=r; steps+=1
            if done or trunc:
                print(f"[BC Eval] Return={ret:.2f} Steps={steps}")
                ret,steps=0,0; obs,_=env.reset()
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
