from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 log_dir='run',
                 device='cpu',
                 shuffle_batch=True):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def observe(self, actor_obs):
        self.actor_obs = actor_obs
        self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        # self.actions = np.clip(self.actions.numpy(), self.env.action_space.low, self.env.action_space.high)
        return self.actions.cpu().numpy()

    def step(self, value_obs, rews, dones):
        values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, rews, dones, values,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device)) # return expected value of last steo
        # last observation is not used, we didnt compute the value for the last observation
        # critic evaluates, value, you compute the value pass it to storage? why? for TD lambda and GAE, we need a value function
        # need to evaluate predicted value from value network to compute lower variance estimates, then compute returns

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.gamma, self.lam) # compute TD-lambda and GAE
        mean_value_loss, mean_surrogate_loss, infos = self._train_step()
        # doing the real training on value function and policy function network

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

        self.storage.clear()


    def log(self, variables, width=80, pad=28): #log data to tensorf board
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()

        self.writer.add_scalar('Loss/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('Loss/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('Loss/average_rewards', torch.mean(self.storage.rewards).detach(), variables['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), variables['it'])

    def _train_step(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        for epoch in range(self.num_learning_epochs):  # for multiple updates PPO, so in this case 4 epochs in runner.py
            # 1 epoch = one sweep of your dataset, which is split into min_batches defined in runner.py,
            # NN is highly non-linear, if you take multiple steps, mitigate non linearity in NN
            # still it will not help with non-linearities (approximation mu prime to mu in PPO, which is valid up to first order)
            # of the policy gradient,
            # if we keep learning speed, decreasing epoch => increasing learning rate, but it ignores the epoch is non-linear
            # so better if we apply smaller step
            for actor_obs_batch, critic_obs_batch, actions_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):  # do the training/compute surrogate loss, for each epoch, each training/epoch, each minibatch
                # See PPO Lecture! actual PPO training
                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch) # input actual action we took
                # output = the log probability of taking that action, entropy of the stochastic policy
                value_batch = self.critic.evaluate(critic_obs_batch)
                # actions_log_prob_batch and value_batch is NOT a number, it is a mathematical expression,
                # like an equation which uses the active parameters, policy parameters.
                # so we can take the derivative of expression

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # importance sampling ratio. old_actions_log_prob_batch is just a number. e^(ln(a)-ln(b)= a/b
                surrogate = -torch.squeeze(advantages_batch) * ratio # rt * At, see LCPI(theta) PPO paper
                # we put negative, pytorch formulate minimization problem, convention
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                # clamp is CLIP function
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() # max because we put negative

                # Value function loss
                if self.use_clipped_value_loss: # see PPO Lecture, #7 value function! not in PPO
                    value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2) # LVF_Clip
                    value_losses = (value_batch - returns_batch).pow(2) # this is exactly MC, LVF
                    value_loss = torch.max(value_losses, value_losses_clipped).mean() # LVF
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                # balance between value learning and policy learning, negative entropy coef, want to maximize the entropy
                # surrogate loss, value loss, entropy batch, are expressions! not numbers! => hence can take derivative
                # surrogate loss and entropy batch depends on theta, value loss depends on w
                # when we run optimizer.step, surrogate loss and entropy loss is not gonna affect value func
                # value lossi is not gonna affect policy func

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()  # compute gradient of loss w.r.t of actor and critic parameters
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
