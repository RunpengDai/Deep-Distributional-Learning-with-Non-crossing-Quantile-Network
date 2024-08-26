import torch
from torch.optim import Adam

from agents.model import DEnet
from agents.utils import disable_gradients, update_params,\
    calculate_quantile_loss, evaluate_quantile_at_action, calculate_quantile_huber_loss
from .base_agent import BaseAgent


class DEnetAgent(BaseAgent):

    def __init__(self, env, test_env, log_dir, num_steps=5*(10**7),
                 batch_size=32, N=200, kappa=1.0, lr=5e-5, memory_size=10**6,
                 gamma=0.99, multi_step=1, update_interval=4,
                 target_update_interval=10000, start_steps=50000,
                 epsilon_train=0.01, epsilon_eval=0.001,
                 epsilon_decay_steps=250000, double_q_learning=False,
                 dueling_net=False, noisy_net=False,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None, cuda=True,
                 seed=0, star = False, exploration=False, tensorboard=False):
        super(DEnetAgent, self).__init__(
            env, test_env, log_dir, num_steps, batch_size, memory_size,
            gamma, multi_step, update_interval, target_update_interval,
            start_steps, epsilon_train, epsilon_eval, epsilon_decay_steps,
            double_q_learning, dueling_net, noisy_net, log_interval,
            eval_interval, num_eval_steps, max_episode_steps, grad_cliping,
            cuda, seed, exploration, tensorboard)

        # Online network.
        self.online_net = DEnet(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, dueling_net=dueling_net,
            noisy_net=noisy_net, star = star).to(self.device)
        # Target network.
        self.target_net = DEnet(
            num_channels=env.observation_space.shape[0],
            num_actions=self.num_actions, N=N, dueling_net=dueling_net,
            noisy_net=noisy_net, star = star).to(self.device).to(self.device)

        # Copy parameters of the learning network to the target network.
        self.update_target()
        # Disable calculations of gradients of the target network.
        disable_gradients(self.target_net)

        self.optim = Adam(
            self.online_net.parameters(),
            lr=lr, eps=1e-2/batch_size)

        # Fixed fractions.
        taus = torch.arange(
            0, N+1, device=self.device, dtype=torch.float32) / N
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, N)

        self.N = N
        self.kappa = kappa

    def learn(self):
        self.learning_steps += 1

        states, actions, rewards, next_states, dones =\
            self.memory.sample(self.batch_size)

        quantile_loss= self.calculate_loss(
            states, actions, rewards, next_states, dones)

        update_params(
            self.optim, quantile_loss,
            networks=[self.online_net],
            retain_graph=False, grad_cliping=self.grad_cliping)


    def calculate_loss(self, states, actions, rewards, next_states, dones):
        current = self.online_net(states=states)
        current_sa_quantiles = evaluate_quantile_at_action(
            current,
            actions)
        assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

        with torch.no_grad():
            # Calculate Q values of next states.
            if self.double_q_learning:
                next_q = self.online_net.calculate_q(states=next_states)
            else:
                next_q = self.target_net.calculate_q(states=next_states)

            # Calculate greedy actions.
            next_actions = torch.argmax(next_q, dim=1, keepdim=True)
            assert next_actions.shape == (self.batch_size, 1)

            # Calculate quantile values of next states and actions at tau_hats.
            next_sa_quantiles = evaluate_quantile_at_action(
                self.target_net(states=next_states),
                next_actions).transpose(1, 2)
            assert next_sa_quantiles.shape == (
                self.batch_size, 1, self.N)

            # Calculate target quantile values.
            target_sa_quantiles = rewards[..., None] + (
                1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
            assert target_sa_quantiles.shape == (self.batch_size, 1, self.N)

        td_errors = target_sa_quantiles - torch.flip(current_sa_quantiles, [-1])
        assert td_errors.shape == (self.batch_size, self.N, self.N)

        quantile_loss = calculate_quantile_huber_loss(td_errors, self.tau_hats, self.kappa)

        return quantile_loss