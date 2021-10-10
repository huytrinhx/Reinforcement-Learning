import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import *
import numpy as np
import random

CONVO_1_OUT = 16
CONVO_1_KERNEL_SIZE = 2
CONVO_2_OUT = 32
CONVO_2_KERNEL_SIZE = 3

class Policy(nn.Module):

    def __init__(self, board_size=3):
        super(Policy, self).__init__()
        #board size
        self.board_size = board_size
        #convolution layers
        self.conv1 = nn.Conv2d(1, CONVO_1_OUT, kernel_size=CONVO_1_KERNEL_SIZE, stride=1, bias=False)
        self.conv2 = nn.Conv2d(CONVO_1_OUT, CONVO_2_OUT, kernel_size=CONVO_2_KERNEL_SIZE, stride=1, bias=False)
        self.size = CONVO_2_KERNEL_SIZE**2 * CONVO_2_OUT
        

        # layers for the policy
        self.fc_action1 = nn.Linear(self.size, self.size//2)
        self.fc_action2 = nn.Linear(self.size//2, self.board_size**2)
        
        # layers for the critic
        self.fc_value1 = nn.Linear(self.size, self.size//4)
        self.fc_value2 = nn.Linear(self.size//4, 1)
        self.tanh_value = nn.Tanh()

        #optimizer
        self.optim = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)

        

        
    def forward(self, x):


        y = F.leaky_relu(self.conv1(x))
        # print(y.size())
        y = F.leaky_relu(self.conv2(y))
        y = y.view(-1, self.size)
        
        
        # the action head
        a = F.leaky_relu(self.fc_action1(y))
        a = self.fc_action2(a)
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        avail = avail.reshape(-1, self.board_size**2)
        
        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)
        
        
        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))
        # print(value)
        return prob.view(self.board_size, self.board_size), value

    def optimize(self, leaf, vterm, logterm):
        #minimize policy loss (defined as predicted winner versus actual winner)
        outcome = leaf.outcome
        loss = torch.sum( (torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss)

