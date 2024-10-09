import torch
import numpy as np


class A3C(torch.nn.Module):
    def __init__(self, state_size, n_actions, gamma=0.99, entropy_scale=0.01, tau=0.95):
        super(A3C, self).__init__()
        self.gamma = gamma
        self.entropy_scale = entropy_scale
        self.tau = tau

        self.conv1 = torch.nn.Conv2d(
            state_size[0], 32, kernel_size=3, stride=2, padding=1
        )
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.conv_shape = self._calculate_conv_shape(state_size)
        # print("Conv shape:", self.conv_shape)

        self.gru = torch.nn.GRUCell(self.conv_shape, 256)
        self.pi = torch.nn.Linear(256, n_actions)
        self.v = torch.nn.Linear(256, 1)

    def _calculate_conv_shape(self, state_size):
        o = self.conv1(torch.zeros(1, *state_size))
        o = self.conv2(o)
        o = self.conv3(o)
        o = self.conv4(o)
        return int(np.prod(o.size()))

    def forward(self, x, hidden_state):
        x = torch.nn.functional.elu(self.conv1(x))
        x = torch.nn.functional.elu(self.conv2(x))
        x = torch.nn.functional.elu(self.conv3(x))
        x = torch.nn.functional.elu(self.conv4(x))
        x = x.view(-1, self.conv_shape)

        hidden_state = self.gru(x, (hidden_state))

        pi = self.pi(hidden_state)
        v = self.v(hidden_state)

        probs = torch.nn.functional.softmax(pi, dim=1)
        dist = torch.distributions.Categorical(probs)  # discrete action space
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hidden_state

    def _calculate_returns(self, rewards, values, done):
        values = torch.cat(values).squeeze()

        if len(values.size()) == 1:  # batch of states
            r = values[-1] * (1 - int(done))
        elif len(values.size()) == 0:  # single state
            r = values * (1 - int(done))

        batch_return = []
        for reward in rewards[::-1]:
            r = reward + self.gamma * r
            batch_return.append(r)

        batch_return.reverse()
        return torch.tensor(batch_return, dtype=torch.float).reshape(values.size())

    def _generalized_advantage_estimation(self, delta_t):
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps - t):
                temp = (self.gamma * self.tau) ** k * delta_t[t + k]
                gae[t] += temp

        return torch.tensor(gae, dtype=torch.float)

    def calculate_loss(
        self,
        next_state,
        hidden_state,
        rewards,
        values,
        log_probs,
        done,
        intrinsic_reward=None,
    ):
        if intrinsic_reward is not None:
            # detach to decouple calculations between two networks
            rewards += intrinsic_reward.detach().numpy()

        returns = self._calculate_returns(rewards, values, done)

        next_value = (
            torch.zeros(1, 1)
            if done
            else self.forward(
                torch.tensor(np.array(next_state), dtype=torch.float), hidden_state
            )[1]
        )

        values.append(next_value.detach())  # detach to prevent backpropagation

        values = torch.cat(values).squeeze()
        log_probs = torch.cat(log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        gae = self._generalized_advantage_estimation(delta_t)

        # squeeze to prevent rank error in case of single sample
        critic_loss = torch.nn.functional.mse_loss(values[:-1].squeeze(), returns)
        actor_loss = -(log_probs * gae).sum()
        entropy_loss = (-log_probs * torch.exp(log_probs)).sum()

        return actor_loss + critic_loss - self.entropy_scale * entropy_loss


if __name__ == "__main__":
    state_size = (4, 42, 42)
    n_actions = 6
    model = A3C(state_size, n_actions)
    x = torch.randn(1, *state_size)
    hidden_state = torch.zeros(1, 256)
    action, v, log_prob, hidden_state = model(x, hidden_state)
    print(action, log_prob, v, hidden_state)
