from typing import Tuple
import numpy as np
import pickle
from tqdm.std import tqdm
from QL.QLearning import QLearning
from SetteEMezzoGame import SetteEMezzo

ALPHA = 0.9


class SetteEMezzoQL(SetteEMezzo, QLearning):
    def __init__(self, n_episodes, eps_start=0.3, lr=0.001, policy=(-1, -1)) -> None:
        QLearning.__init__(self, n_episodes, eps_start, lr)
        SetteEMezzo.__init__(self)

        self.Q_values = self.init_q_values()
        self.state_action = []
        self.done = False
        self.policy = policy

    def init_q_values(self):
        q_values = {}
        for cards_value in np.arange(0.5, 8.0, 0.5):
            for bust_prob in range(0, 105, 5):
                q_values[(cards_value, bust_prob)] = {}
                for action in self.actions:
                    q_values[(cards_value, bust_prob)][action] = 0
        return q_values

    def get_action(self, cards_value_bust_prob: Tuple[float, float]):
        if np.random.uniform(0.1) <= self.eps():
            return super().get_action(cards_value_bust_prob)

        else:
            return max(
                self.Q_values[cards_value_bust_prob],
                key=self.Q_values[cards_value_bust_prob].get,
            )

    def update_Q_values(self, state, action, reward, new_state, is_finished):
        if not is_finished:
            self.Q_values[state][action] = self.Q_values[state][action] + self.lr * (
                reward
                + ALPHA
                * self.Q_values[new_state][
                    max(self.Q_values[new_state], key=self.Q_values[new_state].get)
                ]
                if is_finished
                else 0 - self.Q_values[state][action]
            )
        else:
            self.Q_values[state][action] = self.Q_values[state][action] + self.lr * (
                reward - self.Q_values[state][action]
            )
        # self.Q_values[state][action] = round(reward, 3)

    def play(self):
        self.win_ratio = []
        print("##############    7 e Mezzo Q-Learning    ##############")
        tqdm_ = tqdm(range(0, self.n_episodes))
        for ep in tqdm_:
            self.ep = ep
            self.state_action = []

            state = self.start_new_episode()
            while True:

                action = self.get_action(state)

                new_state, reward, is_finished = self.step(action=action)

                self.update_Q_values(state, action, reward, new_state, is_finished)

                state = new_state
                if is_finished:
                    break

            self.win_ratio.append(self.win / (ep + 1))
            tqdm_.set_description(
                "Ep {}/{}: eps: {} ; win ratio: {}".format(
                    ep + 1,
                    self.n_episodes,
                    round(self.eps(), 3),
                    round(self.win / (ep + 1), 3),
                )
            )
        return self.Q_values, self.win, self.win_ratio

    def evaluate(self):
        self.win_ratio = []
        self.win = 0
        print("##############    7 e Mezzo Q-Learning    ##############")
        tqdm_ = tqdm(range(0, self.n_episodes))
        self.no_eps = True
        self.n_episodes = 100000
        for ep in tqdm_:
            self.ep = ep
            self.run_episode()
            self.win_ratio.append(self.win / (ep + 1))
            tqdm_.set_description(
                "Ep {}/{}: eps: {} ; win ratio: {}".format(
                    ep + 1,
                    self.n_episodes,
                    round(self.eps(), 3),
                    round(self.win / (ep + 1), 3),
                )
            )
        return self.Q_values, self.win, self.win_ratio

    def get_q_table(self):
        return self.Q_values

    def save_policy(self, file="policyQL"):
        fw = open("models/" + file, "wb")
        pickle.dump(self.Q_values, fw)
        fw.close()

    def load_policy(self, file="policyQL"):
        fr = open("models/" + file, "rb")
        self.Q_values = pickle.load(fr)
        fr.close()
