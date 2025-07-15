import torch
import torch.nn as nn

class CollisionPredictor():
    def __init__(self, input_dim=14):
        super(CollisionPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(x)