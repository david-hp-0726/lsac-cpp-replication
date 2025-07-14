import numpy as np
from action_sampler import ActionSampler
import mujoco

class DataCollector:
    def __init__(self, env, history_len=2, delta_T=4):
        self.env = env
        self.history_len = history_len
        self.delta_T = delta_T
        self.buffer = []
    
    def run_episode(self, max_steps=2000):
        # self.env.reset()
        obs_buffer = []
        labels = []
        sampler = ActionSampler()

        for i in range(max_steps):
            action = sampler.sample()
            self.env.step(action)
            
            obs = self.env.get_observation()
            obs_buffer.append(obs)
            labels.append(0)

            if i % 50 == 0:
                print(obs)
            
            if self.env.collided():
                print(f'collided at step {i}')
                i_reverse = i
                counter = self.delta_T
                while i_reverse >= 0 and counter >= 0:
                    labels[i_reverse] = 1
                    counter -= 1
                    i_reverse -= 1
                break
        return {
            'observations': obs_buffer,
            'labels': labels
        }

    
