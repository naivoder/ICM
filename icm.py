import torch
import numpy as np


class ICM(torch.nn.Module):
    def __init__(self, input_dims, n_actions=3, alpha=0.1, beta=0.2):
        super(ICM, self).__init__()
        self.alpha = alpha
        self.beta = beta

        n_channels = 4
        self.conv1 = torch.nn.Conv2d(n_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, 2, 1)

        self.phi = torch.nn.Conv2d(32, 32, 3, 2, 1)
        self.inverse = torch.nn.Linear(288 * 2, 256)

        self.pi = torch.nn.Linear(
            256, n_actions
        )  # policy logits, pass to cross-entropy loss (performs softmax implicitly)

        self.dense1 = torch.nn.Linear(288 + 1, 256)  # feature representation + action
        self.phi_hat = torch.nn.Linear(256, 288)  # predicted feature representation

        device = torch.device("cpu")  # no need for GPU
        self.to(device)

    def forward(self, state, next_state, action):
        x = torch.nn.functional.elu(self.conv1(state))
        x = torch.nn.functional.elu(self.conv2(x))
        x = torch.nn.functional.elu(self.conv3(x))
        phi = torch.nn.functional.elu(self.phi(x))

        y = torch.nn.functional.elu(self.conv1(next_state))
        y = torch.nn.functional.elu(self.conv2(y))
        y = torch.nn.functional.elu(self.conv3(y))
        phi_next = torch.nn.functional.elu(self.phi(y))

        # [T, 32, 3, 3] -> [T, 288]
        phi = phi.view(phi.size()[0], -1).to(torch.float)
        phi_next = phi_next.view(phi_next.size()[0], -1).to(torch.float)

        inverse = self.inverse(torch.cat([phi, phi_next], dim=1))
        pi_logits = self.pi(inverse)

        # [T] -> [T, 1]
        action = action.reshape((action.size()[0], 1)).to(torch.float)
        forward = self.dense1(torch.cat([phi, action], dim=1))
        phi_next_hat = self.phi_hat(forward)

        return phi_next, pi_logits, phi_next_hat

    def calculate_loss(self, states, next_states, actions):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)

        phi_next, pi_logits, phi_next_hat = self.forward(states, next_states, actions)

        inverse_loss = (1 - self.beta) * torch.nn.functional.cross_entropy(
            pi_logits, actions
        )
        forward_loss = self.beta * torch.nn.functional.mse_loss(phi_next_hat, phi_next)

        # very important! want to make sure we get mean for each state, not across all states...
        intrinsic_reward = (
            self.alpha * 0.5 * ((phi_next - phi_next_hat).pow(2)).mean(dim=1)
        )

        return intrinsic_reward, inverse_loss, forward_loss
