from torch import nn
import torch
import numpy as np
import time

import torch.nn.functional as F
from fqf_iqn_qrdqn.network import DQNBase, NoisyLinear

class DEnet(nn.Module):

    def __init__(self, num_channels, num_actions, N=200, embedding_dim=7*7*64,
                 dueling_net=False, noisy_net=False, network = "old"):
        super(DEnet, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        self.network = network
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Feature extractor of DQN.
        self.dqn_net = DQNBase(num_channels=num_channels)
        # Quantile network.
            
        if not dueling_net:
            if network == "new":
                self.q_net = nn.Sequential(
                    linear(embedding_dim, 512),
                    nn.ReLU(),
                    linear(512, num_actions * N),
                )
                for module in self.q_net.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.zeros_(module.weight)
            else:
                self.q_net = linear(512, num_actions * N)
        else:
            self.advantage_net = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, num_actions * N),
            )
            self.baseline_net = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, N),
            )
        self.embed_net = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU())
        
        if network == "new":
            self.vnet = nn.Sequential(
                linear(embedding_dim, 512),
                nn.ReLU(),
                linear(512, 1*num_actions))
        else:
            self.vnet = nn.Sequential(linear(512, 50),
            nn.ReLU(),
            linear(50, 1*num_actions))


        self.N = N
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        embed = self.embed_net(state_embeddings)
        if not self.dueling_net:
            if self.network == "new":
                quantiles = self.q_net(
                    state_embeddings).view(batch_size, self.N, self.num_actions)
            else:
                quantiles = self.q_net(
                    embed).view(batch_size, self.N, self.num_actions)


        else:
            advantages = self.advantage_net(
                state_embeddings).view(batch_size, self.N, self.num_actions)
            baselines = self.baseline_net(
                state_embeddings).view(batch_size, self.N, 1)
            quantiles = baselines + advantages\
                - advantages.mean(dim=2, keepdim=True)

        # (batch_size, self.N, self.num_actions)
        sigma = F.elu(quantiles) + 1
        assert sigma.shape == (batch_size, self.N, self.num_actions)


        if self.network == "new":
            value = self.vnet(state_embeddings).view(batch_size, 1, self.num_actions)
        else:
            value = self.vnet(embed).view(batch_size, 1, self.num_actions)

        accu = torch.cumsum(sigma, dim=1)

        nc_quantiles =  value + accu - accu.mean(dim=1, keepdim=True)

        assert nc_quantiles.shape == (batch_size, self.N, self.num_actions)

        return nc_quantiles

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None\
            else state_embeddings.shape[0]

        # Calculate quantiles.
        quantiles = self(states=states, state_embeddings=state_embeddings)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q

    def reset_noise(self):
        if self.noisy_net:
            for m in self.modules():
                if isinstance(m, NoisyLinear):
                    m.reset_noise()
