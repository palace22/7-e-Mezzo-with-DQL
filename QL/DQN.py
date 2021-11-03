from torch import nn


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
