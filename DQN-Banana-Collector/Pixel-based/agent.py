import numpy as np
import random

from model import QNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.95
TAU  = 0.06
LR = 5e-4
UPDATE_EVERY = 10
STACK_SIZE = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    
    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        #Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        #Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        #Initialize time step ( for updating every UPDATE_EVERY steps)
        self.t_step = 0
        #Implement stack image buffer
        self.stack_size = STACK_SIZE

    def preprocess_state(self, img):
        return img @ (0.3,0.1,0.7)

    def stack_images(self, img1, img2):
        # https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        # if image is in greyscale and img1 is fully-stacked
        # https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/Chapter%2011/DQN_Atari_v2.ipynb
        # print(img1.shape)
        if img1.shape == 3 and img1.shape[0] == self.stack_size:
            im = np.append(img1[1:,:,:], np.expand_dims(img2,0), axis=2)
            im = np.expand_dims(im,axis=0)
            # print(im.shape)
            return im
        else: #otherwise, clone img1 to the size of the stack hyperparams
            im = np.vstack([img1]*self.stack_size)
            im = np.squeeze(im,axis=None)
            # print(im.shape)
            return im




    def step(self, state, action, reward, next_state, done):
        # Save experiece in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #if enough samples are available in memory, get randome subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        #epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

        
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def save_checkpoints(self, model_dir="./model_dir/", mean_score=0.0):  
        torch.save(self.qnetwork_local.state_dict(), "{0}agent_checkpoint_{1}.pth".format(model_dir,int(mean_score)))

    def load_checkpoints(self, model_dir="./model_dir/", mean_score=0.0):
        self.qnetwork_local.load_state_dict(torch.load("{0}agent_checkpoint_{1}.pth".format(model_dir,int(mean_score))))

        
class DQNAgent_PER(DQNAgent):

    def __init__(self, state_size, action_size, seed):
        super(DQNAgent_PER, self).__init__(state_size, action_size, seed)
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idxs, weights = experiences
       

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss MSE
        loss = (Q_expected - Q_targets.detach()).pow(2)
        # Add weights to loss
        loss = loss * weights
        # Add noise to loss to arrive at prior weights
        prios = loss + 1e-6
        # Take mean
        loss = loss.mean()

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update buffer priorities
        self.memory.update_priorities(zip(idxs, prios.data.cpu().numpy()))



        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     





