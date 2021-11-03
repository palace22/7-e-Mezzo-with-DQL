from typing import Deque
from random import sample


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = Deque([], maxlen=capacity)

    def save(self, obs):
        self.buffer.append(obs)

    def get_batch(self, dim=256):
        return sample(self.buffer, dim)

    def __len__(self):
        return len(self.buffer)
