import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            stack_size (int) : Number of recent frames stacked with current frame
            seed (int): Random seed
            1st conv: 32 filters of 8x8 with stride 4 + nonlinear rectifier
            2nd conv: 64 filtets of 4x4 with stride 2 + nonlinear rectifier
            3rd conv: 64 filters of 3x3 with stride 1 + nonlinear rectifier
            4rd fc: 512 fully-connected nodes 
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=4,out_channels=32,kernel_size=8,stride=4)
        # self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 4)
        self.conv3 = nn.Conv2d(64, 64, 2, 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 1)

        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x1 = F.relu(self.conv1(state))
        x2 = self.pool2(F.relu(self.conv2(x1)))
        x3 = self.pool3(F.relu(self.conv3(x2)))
        #prep for linear layer by flattening feature maps into feature vector        
        x3 = x3.view(x3.size(0), -1)
        #linear layer
        x4 = F.relu(self.fc1(x3))    
        return F.relu(self.fc2(x4))
