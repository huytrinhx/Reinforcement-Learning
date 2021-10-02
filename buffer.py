from collections import deque, namedtuple
import random
import torch
import numpy as np
from utilities import transpose_list

device = "cpu"

class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def push(self,transition):
        """push into the buffer"""
        
        # input_to_buffer = transpose_list(transition)

        # print("\nAfter transpose to list")
        # print(input_to_buffer)
    
        self.deque.append(transition)

    def sample(self, batchsize):
        """sample from the buffer"""
        experiences = random.sample(self.deque, batchsize)
        
        return transpose_list(experiences)

    def __len__(self):
        return len(self.deque)



