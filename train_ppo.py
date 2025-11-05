import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from hexapod_env import HexapodEnv

LOGDIR = "./logs_hexapod"
os.makedirs(LOGDIR, exist_ok=True)

def make_env():
    return HexapodEnv(
        xml_path="hexapod.xml",
        render_mode="none",
        action_repeat=20,
        target_speed=0.35,
        obs_clip=10.0,
        seed=42
    )
env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[96, 96]),
    learning_rate = 3e-4,
    n_steps = 4000, # 4000 transitions / 1 update
    batch_size = 512,
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    ent_coef = 0.001,
    vf_coef = 0.5,
    verbose = 1,

)

new_logger = configure(LOGDIR, ["tensorboard", "stdout"])
model.set_logger(new_logger)

model.learn(total_timesteps=500_000) #정책 업데이트 수 : total_timesteps / n_steps -> total_timesteps를 최소 500만 이상으로 할 것.
model.save(os.path.join(LOGDIR, "ppo_hexapod"))

test_env = HexapodEnv(xml_path="hexapod.xml", render_mode="human", action_repeat=20, target_speed=0.35)
obs, _ = test_env.reset()
for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = test_env.step(action)
    if term or trunc:
        obs, _ = test_env.reset()
test_env.close()