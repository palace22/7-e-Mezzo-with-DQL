from typing import Deque
from random import sample
from matplotlib import pyplot as plt
import numpy as np
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = Deque([], maxlen=capacity)

    def save(self, obs):
        self.buffer.append(obs)

    def get_batch(self, dim=256):
        return sample(self.buffer, dim)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        k = 256
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, 2),
        )

    def forward(self, x):
        x[:, 0] = (x[:, 0] - (7.5 / 2)) / 7.5
        x[:, 1] = (x[:, 1] - (50)) / 100
        logits = self.linear_relu_stack(x)
        return logits


def eps(ep):
    eps_start = 0.4
    eps_end = 0.001
    n_episodes = 250000
    eps_decay = int(np.ceil(n_episodes / 3))  # /3
    decay_ep = n_episodes - n_episodes / 5
    no_eps = False
    return eps_end + max(
        (eps_start - eps_end) * (1 - np.exp((ep - decay_ep) / eps_decay)),
        0,
    )


data = [eps(i) for i in range(0, 250000)]

plt.title("eps")
plt.xlabel("ep")
plt.ylabel("eps")
plt.plot([i for i in range(0, len(data))], data)
plt.show()
