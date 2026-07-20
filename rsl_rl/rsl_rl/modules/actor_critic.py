# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        self.n = 1e-4
        self.uninitialized = True
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)

    def update(self, x):
        count = self.n
        batch_count = x.size(0)
        tot_count = count + batch_count

        old_mean = self.mean.clone()
        delta = torch.mean(x, dim=0) - old_mean

        self.mean = old_mean + delta * batch_count / tot_count
        m_a = self.var * count
        m_b = x.var(dim=0) * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        self.var = M2 / tot_count
        self.n = tot_count

class Normalization:
    def __init__(self, shape, device='cuda:0'):
        self.running_ms = RunningMeanStd(shape=shape, device=device)

    def __call__(self, x, update=False):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (torch.sqrt(self.running_ms.var) + 1e-4)

        return x

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,
                num_actor_obs,
                num_critic_obs,
                num_one_step_obs,
                actor_history_length,
                num_actions=19,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
                init_noise_std=1.0,
                **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_one_step_obs = num_one_step_obs

        self.actor_history_length = actor_history_length

        self.num_actions = num_actions

        self.history_latent_dim = 16

        self.estimate_ball_dim = 6

        self.num_regions = 6

        self.history_obs_dim = num_one_step_obs * actor_history_length
        self.task_cue_dim = num_actor_obs - self.history_obs_dim
        self.expected_task_cue_dim = 3 + 1 + 1 + 1
        if self.task_cue_dim != self.expected_task_cue_dim:
            raise ValueError(
                f"ActorCritic expects {self.expected_task_cue_dim} task cue values "
                f"after the {self.history_obs_dim}-value history, got {self.task_cue_dim}"
            )

        mlp_input_dim_a = (
            num_one_step_obs
            + self.history_latent_dim
            + self.estimate_ball_dim
            + 1  # region ID
            + 1  # launch flag
            + 1  # estimator-ready flag
        )
        
        self.num_actor_input  = mlp_input_dim_a

        mlp_input_dim_c = num_critic_obs

        mlp_input_dim_h = self.history_obs_dim

        self.history_encoder = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.history_latent_dim),
        )

        self.ball_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.estimate_ball_dim),
        )

        self.region_estimator = nn.Sequential(
            nn.Linear(mlp_input_dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_regions),
        )



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)


        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"History MLP: {self.history_encoder}")
        print(f"Ball MLP: {self.ball_estimator}")
        print(f"Region MLP: {self.region_estimator}")
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.estimate_ball = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs_history):
        actor_input, self.estimate_ball, self.estimate_region = self._actor_input(obs_history)
        
        action_mean = self.actor(actor_input)
        
        self.distribution = Normal(action_mean, action_mean*0. + self.std)

    def act(self, obs_history=None, **kwargs):
        self.update_distribution(obs_history)
        return self.distribution.sample(), self.estimate_ball, self.estimate_region
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs_history, observations=None):
        actor_input, _, _ = self._actor_input(obs_history)

        action_mean = self.actor(actor_input)

        return action_mean

    def _actor_input(self, observations):
        history = observations[:, :self.history_obs_dim]
        cue = observations[:, self.history_obs_dim:]

        prior_target = cue[:, :3]
        prior_region_id = cue[:, 3:4]
        launch_flag = cue[:, -2:-1].clamp(0.0, 1.0)
        estimator_ready = cue[:, -1:].clamp(0.0, 1.0)

        history_latent = self.history_encoder(history)
        estimate_ball_raw = self.ball_estimator(history)
        estimate_region_logits = self.region_estimator(history)
        estimate_region_id = torch.argmax(
            estimate_region_logits, dim=-1, keepdim=True
        ).to(dtype=estimate_ball_raw.dtype)
        prior_region_id = prior_region_id.to(dtype=estimate_ball_raw.dtype)

        prior_ball = torch.cat(
            (prior_target, torch.zeros_like(estimate_ball_raw[:, 3:])), dim=-1
        )
        ball_used = (
            (1.0 - estimator_ready) * prior_ball
            + estimator_ready * estimate_ball_raw
        )
        region_used = (
            (1.0 - estimator_ready) * prior_region_id
            + estimator_ready * estimate_region_id
        )

        actor_input = torch.cat(
            (
                history[:, -self.num_one_step_obs:],
                history_latent,
                ball_used,
                region_used,
                launch_flag,
                estimator_ready,
            ),
            dim=-1,
        )
        return actor_input, estimate_ball_raw, estimate_region_logits


    def evaluate(self, critic_observations, **kwargs):
        
        value = self.critic(critic_observations)

        return value
