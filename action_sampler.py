import numpy as np

class ActionSampler:
    def __init__(self,
                 fwd_range=(-0.6, 1.0),
                 turn_range=(-1.0, 1.0),
                 drift_interval=100,
                 drift_strength=0.6,
                 fwd_momentum=0.85,
                 turn_momentum=0.75,
                 fwd_noise_scale=0.2,
                 turn_noise_scale=0.2):
        self.fwd_range = fwd_range
        self.turn_range = turn_range

        self.drift_interval = drift_interval
        self.drift_strength = drift_strength

        # Independent momentum allows for more realistic movements
        self.fwd_momentum = fwd_momentum
        self.turn_momentum = turn_momentum
        self.fwd_noise_scale = fwd_noise_scale
        self.turn_noise_scale = turn_noise_scale

        self.step_count = 0
        self.fwd = np.random.uniform(*fwd_range)
        self.turn = 0.0
        self._sample_new_turn_bias()

    def _sample_new_turn_bias(self):
        # The bias is held constant for a while so the car stays committed to a direction
        # The momentum makes the car gradually change direction or speed, instead of constantly wiggling
        self.turn_bias = np.random.choice([-1.0, 1.0]) * self.drift_strength
        # print(f"[Step {self.step_count}] Sampled new turn_bias = {self.turn_bias:.2f}")

    def sample(self):
        if self.step_count % self.drift_interval == 0:
            self._sample_new_turn_bias()

        # Forward update
        fwd_noise = np.random.uniform(-1, 1)
        self.fwd = self.fwd_momentum * self.fwd + self.fwd_noise_scale * fwd_noise
        self.fwd = np.clip(self.fwd, *self.fwd_range)

        # Turn update
        turn_noise = np.random.uniform(-1, 1)
        self.turn = self.turn_momentum * self.turn + (1 - self.turn_momentum) * self.turn_bias + self.turn_noise_scale * turn_noise
        self.turn = np.clip(self.turn, *self.turn_range)

        self.step_count += 1
        return np.array([self.fwd, self.turn])
