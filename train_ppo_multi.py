import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from hexapod_env import HexapodEnv

LOGDIR = "./logs_hexapod"
XML_PATH = "hexapod_uneven.xml"
N_ENVS = 4

# 옵션: CPU 스레드 제한(발열/스로틀링 완화)
torch.set_num_threads(4)
os.environ.setdefault("OMP_NUM_THREADS", "4")

def make_env(rank, seed=42):
    # ⚠️ 함수는 모듈 최상위에 있어야(피클 가능) 하며,
    # SubprocVecEnv는 이 팩토리를 워커 프로세스에서 호출합니다.
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

    # 진짜 병렬
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecMonitor(venv, filename=os.path.join(LOGDIR, "monitor.csv"))

    # 총 rollout ~= 4096/업데이트 (1024 x 4)
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=[96, 96]),
        learning_rate=3e-4,
        n_steps=1024,        # per-env steps → 4*1024 = 4096
        batch_size=1024,     # 4096의 약수(512도 OK)
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

    model.learn(total_timesteps=10_000_000)
    model.save(os.path.join(LOGDIR, "ppo_hexapod"))
    venv.close()

    # ===== 별도 평가(단일 env + 렌더) =====
    eval_env = HexapodEnv(XML_PATH, render_mode="human", action_repeat=20, target_speed=0.35)
    obs, _ = eval_env.reset()
    model = PPO.load(os.path.join(LOGDIR, "ppo_hexapod"))

    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = eval_env.step(action)
        if done or trunc:
            obs, _ = eval_env.reset()
    eval_env.close()

if __name__ == "__main__":
    import multiprocessing as mp
    # macOS/Windows에서는 spawn이 안전
    mp.set_start_method("spawn", force=True)
    main()
