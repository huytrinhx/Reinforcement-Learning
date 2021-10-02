# individual network settings for each actor + critic pair
# see networkforall for details

from networkforall import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

device = 'cpu'

class DDPGAgent:
    def __init__(self, seed, in_actor, hidden_in_actor_1, hidden_out_actor, out_actor, in_critic, hidden_in_critic_1, hidden_out_critic, lr_actor=1.0e-4, lr_critic=1.0e-3, decay=0):
        super(DDPGAgent, self).__init__()

        self.seed = seed
        self.actor = Network(self.seed,in_actor, hidden_in_actor_1, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(self.seed,in_critic, hidden_in_critic_1, hidden_out_critic, 1).to(device)
        self.target_actor = Network(self.seed,in_actor, hidden_in_actor_1, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(self.seed,in_critic, hidden_in_critic_1,hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor,scale=1)

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=decay)


    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs) + noise * self.noise.noise()
        self.actor.train()
        return np.clip(action,-1,1)

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        self.target_actor.eval()
        with torch.no_grad():
            action = self.target_actor(obs) + noise * self.noise.noise()
        self.target_actor.train()
        return np.clip(action,-1,1)

    def reset(self):
        self.noise.reset()