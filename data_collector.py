import numpy as np
from action_sampler import ActionSampler
import mujoco

class DataCollector:
    def __init__(self, env, history_len=2, delta_T=4):
        self.env = env
        self.history_len = history_len
        self.delta_T = delta_T
        self.buffer = []
    
    def run_episode(self, max_steps=1000):
        obs_buffer = []
        labels = []
        sampler = ActionSampler()
        self.env.reset()

        for i in range(max_steps):
            action = sampler.sample()
            self.env.step(action)
            
            obs = self.env.get_observation()
            obs_buffer.append(obs)
            labels.append(0)

            # if i == 0:
            #     obs_buffer[0] = np.concatenate([obs_buffer[0], [0, 0]])
            # else:
            #     obs_buffer[i] = np.concatenate([obs_buffer[i], obs_buffer[i-1][-4:-2]]) # extend with previous action

            
            if self.env.collided():
                # print(f'collided at step {i}')
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

    
