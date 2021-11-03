from collections import defaultdict
from typing import Tuple
import numpy as np
from tqdm.std import tqdm
from SetteEMezzoGame import SetteEMezzo


class SetteEMezzoMC(SetteEMezzo):
    def __init__(self, n_episodes, policy=(-1, -1)):
        super().__init__()
        self.n_episodes = n_episodes
        self.policy = policy

    def play(self):
        self.win_ratio = []
        self.discount_factor = 0.8
        value_table = defaultdict(float)
        states_count = defaultdict(int)
        value_count = defaultdict(float)
        for cards_value in np.arange(0.5, 8.0, 0.5):
            for bust_prob in range(0, 100, 5):
                value_table[(cards_value, bust_prob)] = 0

        tqdm_ = tqdm(range(0, self.n_episodes))
        print("##############    7 e Mezzo Monte Carlo    ##############")
        for ep in tqdm_:
            states, actions, rewards = self.run_episode()

            for i, state in enumerate(states):
                v_m = (self.discount_factor ** i) * rewards[i]

                value_count[state] += v_m
                states_count[state] += 1
                value_table[state] = value_count[state] / states_count[state]

            self.win_ratio.append(self.win / (ep + 1))
            tqdm_.set_description(
                "Ep {}/{}; win ratio: {}".format(
                    ep + 1,
                    self.n_episodes,
                    round(self.win / (ep + 1), 3),
                )
            )
        return value_table, self.win, self.win_ratio

    def evaluate(self):
        pass

    def get_q_table(self):
        q_table = {}
        for value in np.arange(0.5, 8, 0.5):
            for bust_prob in np.arange(0, 105, 5):
                q_table[(value, bust_prob)] = {}
                v = self.get_action((value, bust_prob))
                q_table[(value, bust_prob)][0] = (v + 1) % 2
                q_table[(value, bust_prob)][1] = v % 2
        return q_table
