import mujoco
import mujoco.viewer
import time
import numpy as np
from data_collector import DataCollector
from car_env import CarEnv
from action_sampler import ActionSampler
from collision_predictor import CollisionPredictor
import torch
import os

model = mujoco.MjModel.from_xml_path('safe_drive_env.xml')
data = mujoco.MjData(model)
brick_names = ['brick1', 'brick2', 'brick3', 'brick4']
box_names = ['rand_box1', 'rand_box2', 'rand_box3']
wall_names = ['wall_left', 'wall_right', 'wall_bottom', 'wall_top']

def simulate(log_prob=False):
    car_env = CarEnv()
    sampler = ActionSampler()
    car_env.launch_viewer()
    car_env.reset()  
    time.sleep(10)

    if log_prob:
        model = CollisionPredictor()
        model.load_state_dict(torch.load("model/best_collision_predictor.pth"))
        model.eval()

    for i in range(2000):
        action = sampler.sample()
        car_env.step(action=action)

        if log_prob and i % 5 == 0:
            obs = car_env.get_observation()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits = model(obs_tensor)
                prob = torch.sigmoid(logits).item()
            print(f"[Prediction] Collision probability: {prob:.4f}")

        time.sleep(0.02)
        if car_env.done():
            break

def generate_dataset(dataset_size=100000):
    car_env = CarEnv()
    data_collector = DataCollector(car_env, delta_T=6)
    all_observations = []
    all_labels = []
    episode = 0

    while len(all_labels) < dataset_size:
        data = data_collector.run_episode()
        all_observations.extend(data['observations'])
        all_labels.extend(data['labels'])

        if episode % 100 == 0:
            print(f"Episode {episode} finished in {len(data['observations'])} steps. Currently at {len(all_labels)} examples")
        episode += 1

    print(len(all_labels))
    X = np.array(all_observations[-dataset_size:])
    Y = np.array(all_labels[-dataset_size:])
    np.save('data/X.npy', X)
    np.save('data/Y.npy', Y)

def main():
    # generate_dataset(dataset_size=272000)
    simulate(log_prob=True)

if __name__ == '__main__':
    main()