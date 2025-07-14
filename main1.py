import numpy as np
import time
from safe_drive_env import SafeDriveEnv


def main():
    env = SafeDriveEnv(xml_path='safe_drive_env.xml', render_mode="human")
    obs, _ = env.reset()
    total_reward = 0

    for step in range(10000):
        action = env.action_space.sample()
        action[0] = abs(action[0])

        if step % 1000 == 0:
            print(f"action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

        time.sleep(0.02)

        if terminated:
            print(f"Episode completes at step {step}. Total reward: {total_reward}")
            total_reward = 0
            obs, _ = env.reset()

    
    env.close()

if __name__ == "__main__":
    main()