import numpy as np
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            # create a new spot
            self.buffer.append(None)

        self.buffer[self.position] = (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
            torch.tensor([done], dtype=torch.float32)
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(torch.stack, zip(*batch))
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.buffer)