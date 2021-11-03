from collections import deque
import numpy as np
import torch
from torch import optim
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from QL.DQN import DQN
from QL.ReplayBuffer import ReplayBuffer
from SetteEMezzoGame import SetteEMezzo
import matplotlib.pyplot as plt
from QL.QLearning import QLearning

OPTIMIZE_COUNTER = 10
UPDATE_TARGET_NET_COUNTER = 1000
BATCH_SIZE = 256
REPLAY_BUFFER_CAPACITY = 20000
GAMMA = 0.999
WEIGHT_DECAY = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SetteEMezzoDQN(SetteEMezzo, QLearning):
    def __init__(self, n_episodes, eps_start=0.3, lr=0.0001, policy=(-1, -1)) -> None:
        QLearning.__init__(self, n_episodes, eps_start, lr)
        SetteEMezzo.__init__(self)

        self.replayBuffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

        self.policy_net = DQN()
        self.policy_net.to(device=device)
        self.target_net = DQN()
        self.target_net.to(device=device)
        self.policy_net.eval()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.player_bust_reward = -1
        self.policy = policy

    def get_action(self, cards_value_bust_prob):
        if np.random.uniform(0.1) <= self.eps():
            return super().get_action(cards_value_bust_prob)

        state = (
            torch.tensor([list(cards_value_bust_prob)])
            .type(torch.FloatTensor)
            .to(device=device)
        )

        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def run_episode(self):
        state = self.start_new_episode()
        while True:
            action = self.get_action(state)
            (new_state, reward, is_finished) = self.step(
                action, self.player_bust_reward
            )

            if is_finished:
                new_state = None
                self.replayBuffer.save((state, action, reward, new_state))
                break

            self.replayBuffer.save((state, action, reward, new_state))
            state = new_state

        return reward

    def optimize(self):
        if len(self.replayBuffer) < BATCH_SIZE:
            return -1

        batch_sample = self.replayBuffer.get_batch()
        states, actions, rewards, new_states = [list(x) for x in zip(*batch_sample)]
        states, actions, rewards = (
            torch.tensor(states).to(device=device),
            torch.tensor(actions).to(device=device),
            torch.tensor(rewards).to(device=device),
        )

        self.policy_net.train(True)
        Q_policy = self.policy_net(states).gather(1, actions.view(-1, 1)).squeeze()

        non_final_state_mask = torch.tensor(
            tuple(map(lambda s: s is not None, new_states)), dtype=torch.bool
        )

        non_final_new_states = (
            torch.tensor([s for s in new_states if s is not None])
            .type(torch.FloatTensor)
            .to(device=device)
        )
        with torch.no_grad():
            Q_target = torch.zeros(BATCH_SIZE).type(torch.FloatTensor).to(device=device)
            Q_target[non_final_state_mask] = self.target_net(non_final_new_states).max(
                1
            )[0]

        Y = (Q_target * GAMMA) + rewards

        loss = F.smooth_l1_loss(Q_policy, Y)

        self.optimizer.zero_grad()
        l = loss.item()
        loss.backward()

        self.optimizer.step()
        self.policy_net.train(False)
        return l

    def play(self):
        self.win_ratio = []
        self.win_ma = deque(np.zeros(1000, dtype="int"), maxlen=1000)
        self.win_ma_arr = []
        loss = 0
        tot_rew = 0
        self.player_bust_reward = -1.5

        tqdm_ = tqdm(range(self.n_episodes))

        print("##############    7 e Mezzo Deep Q-Learing    ##############")

        for ep in tqdm_:
            self.ep = ep
            rew = self.run_episode()
            tot_rew += rew
            if ep % OPTIMIZE_COUNTER == 0:
                loss = self.optimize()

            if ep % UPDATE_TARGET_NET_COUNTER == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.win_ratio.append(self.win / (ep + 1))
            self.win_ma.append(rew)
            self.win_ma_arr.append(sum(self.win_ma) / 1000)

            tqdm_.set_description(
                "Ep {}/{}: loss {}; rew: {}; eps: {} ; win ratio: {}".format(
                    ep + 1,
                    self.n_episodes,
                    round(loss, 3),
                    tot_rew,
                    round(self.eps(), 3),
                    round(self.win / (ep + 1), 3),
                )
            )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(self.win_ratio[len(self.win_ratio) - 1])
        print(tot_rew)
        self.plot(self.win_ratio, "Win ratio", "number of episodes", "win ratio")
        self.plot(
            self.win_ma_arr, "Reward moving average (1000 ep.)", "Episodes", "Reward"
        )
        return self.get_q_table(), self.win, self.win_ratio

    def evaluate(self):
        self.no_eps = True
        self.win = 0
        self.win_ratio = []
        self.ep = 0
        self.player_bust_reward = -1
        tot_reward = 0
        self.policy_net.load_state_dict(self.target_net.state_dict())
        self.policy_net.eval()
        self.n_episodes = 250000
        tqdm_ = tqdm(range(self.n_episodes))
        for ep in tqdm_:
            rew = self.run_episode()
            tot_reward += rew
            self.win_ratio.append(self.win / (ep + 1))
            tqdm_.set_description(
                "Loss at ep {}/{}: rew: {}; eps: {} ; win ratio: {}".format(
                    ep + 1,
                    self.n_episodes,
                    tot_reward,
                    round(self.eps(), 3),
                    round(self.win / (ep + 1), 4),
                )
            )

        print(self.win_ratio[len(self.win_ratio) - 1])
        print(self.win / self.n_episodes)
        self.plot(self.win_ratio, "Win ratio", "number of episodes", "win ratio")

        return self.get_q_table(), self.win, self.win_ratio

    def get_q_table(self):
        q_table = {}
        for value in np.arange(0.5, 8, 0.5):
            for bust_prob in np.arange(0, 105, 5):
                q_table[(value, bust_prob)] = {}
                v = self.target_net(
                    torch.tensor([[value, bust_prob]])
                    .type(torch.FloatTensor)
                    .to(device=device)
                )
                q_table[(value, bust_prob)][0] = v[0][0].item()
                q_table[(value, bust_prob)][1] = v[0][1].item()
        return q_table

    def get_q_table_policy(self):
        q_table = {}
        for value in np.arange(0.5, 8, 0.5):
            for bust_prob in np.arange(0, 105, 5):
                q_table[(value, bust_prob)] = {}
                v = self.policy_net(
                    torch.tensor([[value, bust_prob]])
                    .type(torch.FloatTensor)
                    .to(device=device)
                )
                q_table[(value, bust_prob)][0] = v[0][0].item()
                q_table[(value, bust_prob)][1] = v[0][1].item()
        return q_table

    def save_policy(self, policy_net="policy_net.pt", target_net="target_net.pt"):
        torch.save(self.policy_net.state_dict(), "models/" + policy_net)
        torch.save(self.target_net.state_dict(), "models/" + target_net)
        print("Models saved")

    def load_policy(self, policy_net="policy_net.pt", target_net="target_net.pt"):
        self.policy_net.load_state_dict(torch.load("models/" + policy_net))
        self.target_net.load_state_dict(torch.load("models/" + target_net))
        print("Models loaded")

    def plot(self, data, title, x_label, y_label):
        start = int(len(data) / 10)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot([i for i in range(0, len(data[start:]))], data[start:])
        plt.show()
