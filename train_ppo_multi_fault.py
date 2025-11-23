import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList  # ⬅️ 추가
from hexapod_env_fault import HexapodEnv

LOGDIR = "./logs_hexapod"
XML_PATH = "hexapod_uneven.xml"
N_ENVS = 4

# 몇 스텝마다 저장할지 (전체 timesteps 기준)
CHECKPOINT_EVERY = 1_000_000   # <- 원하면 바꿔줘

torch.set_num_threads(4)
os.environ.setdefault("OMP_NUM_THREADS", "4")

def make_env(rank, seed=42):
    def _init():
        env = HexapodEnv(
            xml_path=XML_PATH,
            render_mode="none",
            action_repeat=20,
            target_speed=0.35,
            obs_clip=10.0,
            seed=seed + rank,
        )
        return Monitor(env)
    return _init

def main():
    os.makedirs(LOGDIR, exist_ok=True)
    ckpt_dir = os.path.join(LOGDIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecMonitor(venv, filename=os.path.join(LOGDIR, "monitor.csv"))

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[96, 96]),
        learning_rate=3e-4,
        n_steps=1024,        # per-env steps → 4*1024 = 4096
        batch_size=1024,     # 4096의 약수
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        verbose=1,
    )

    logger = configure(LOGDIR, ["tensorboard", "stdout"])
    model.set_logger(logger)

    # === 체크포인트 콜백: 지정한 스텝마다 저장 ===
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=ckpt_dir,
        name_prefix="ppo_hexapod_fault",   # 파일명: ppo_hexapod_fault_{num_timesteps}_steps.zip
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList([checkpoint_cb])

    model.learn(total_timesteps=30_000_000, callback=callbacks)

    # 최종본 저장
    model.save(os.path.join(LOGDIR, "ppo_hexapod_fault_final"))
    venv.close()

    # ===== 별도 평가(단일 env + 렌더) =====
    eval_env = HexapodEnv(XML_PATH, render_mode="human", action_repeat=20, target_speed=0.35)
    obs, _ = eval_env.reset()
    model = PPO.load(os.path.join(LOGDIR, "ppo_hexapod_fault_final"))

    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = eval_env.step(action)
        if done or trunc:
            obs, _ = eval_env.reset()
    eval_env.close()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
