from collections import namedtuple, deque
from sum_tree import SumTree
import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
PrioritizedExperience = namedtuple('Experience',
                                   ['state', 'action', 'reward', 'next_state', 'done', 'sampling_prob', 'idx'])

class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience
        self.seed = random.seed(seed)
        
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
        
    
    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implementation of proportional prioritization
    Use sum-tree data structure: parent is the sum of all children
    Leaf nodes store the transition priorities and in the internal nodes are intermediate sums
    Parent node contain sum over all priorities
    https://arxiv.org/pdf/1511.05952.pdf

    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        self.tree = SumTree(buffer_size)
        self.max_priority = 1
        
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        self.tree.add(self.max_priority,None)
        
    def sample(self):

        segment = self.tree.total() / self.batch_size
        """"""
        experiences = []
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a,b)
            (idx, p, data_index) = self.tree.get(s)
            e = self.memory[data_index]
            if e is None:
                continue
            experiences.append(PrioritizedExperience(
                *e,
                sampling_prob= p / self.tree.total(),
                idx=idx))
        

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        idxs = torch.from_numpy(np.vstack([e.idx for e in experiences if e is not None])).long().to(device)
        weights = torch.from_numpy(np.vstack([e.sampling_prob for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, idxs, weights)
        
    def update_priorities(self, info):
        for idx, priority in info:
            self.tree.update(idx,priority)
        