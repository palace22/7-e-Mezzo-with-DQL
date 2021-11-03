import numpy as np


class QLearning:
    def __init__(self, n_episodes, eps_start=0.3, lr=0.001) -> None:
        self.n_episodes = n_episodes
        self.lr = lr
        self.start_ep = 0
        self.eps_start = eps_start
        self.eps_end = 0.001
        self.eps_decay = int(np.ceil(n_episodes))  # /3
        self.decay_ep = self.n_episodes - self.start_ep  # - self.n_episodes / 5
        self.no_eps = False

    def eps(self):
        if self.no_eps:
            return -1
        return self.eps_end + max(
            (self.eps_start - self.eps_end)
            * (1 - np.exp((self.ep - self.start_ep - self.decay_ep) / self.eps_decay)),
            0,
        )
