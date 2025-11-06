import time
import os
from stable_baselines3 import PPO
from hexapod_env import HexapodEnv   # 네가 만든 환경
import numpy as np

MODEL_PATH = "./logs_hexapod/ppo_hexapod.zip"   # 저장된 모델 경로

script_dir = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(script_dir, "hexapod.xml")
    
print(f"Loading XML from: {XML_PATH}") # 경로 확인용

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    # 테스트 환경 (렌더 모드 활성화)
    env = HexapodEnv(
        xml_path=XML_PATH,
        render_mode="human",        # 시각화
        action_repeat=20,
        target_speed=0.35,
        obs_clip=10.0,
        seed=0
    )

    obs, _ = env.reset()
    print("Simulation started. Press Ctrl+C to exit.")
    try:
        for step in range(50000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(10)
            time.sleep(0.01)   # 속도 조절용
            if terminated or truncated:
                print("Episode ended. Resetting...")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    env.close()


if __name__ == "__main__":
    main()
