[//]: # (Image References)

[image1]: https://github.com/huytrinhx/DQN-Banana-Collector/blob/main/Images/ScoresChart.png

# Project 2: DDPG for Continuous Actions Environment

### Overview

Using Deep Deterministic Policy Gradient (DDPG) Network, the agent showed it can solve the environment in less than 200 episodes in distributed learning environment. As mentioned in the ReadMe.md, the environment is considered solved if the agent's mean scores in the last 100 episodes is greater than 30. The chart below showed the agent score progression during the training

![Score Chart][image1]

### Algorithms

The full algorithm for training DDPG is fully explained in section Algorithm of the DDPG paper (https://arxiv.org/abs/1509.02971). In the beginning, we initialize the critic network (Q) and actor network (Mu) with random weights. At the same time, we initilize target networks by cloning both critic and actor network.

For each time step in the episode, the agent selects and executes actions according to a policy generated from local actor network and exploration noise. The agent observes reward and new state and store this transition into experience replay buffer . Experience replay means we store the agent's experiences at each time step for many episodes. 

Then, at the same time, we sample the experiences randomly to update the critic by minimizing the loss from predicted actions and values generated from target actor and target critic respectively. Meanwhile, the actor is updated by minimizing the loss from predicted actions and values generated from local actor and local critic respectively.

Another feature of the algorithm is to use delayed intensive update to sync target with the local networks. Particularly, for every learn step (which happened once every k time steps), we blend or soft update the local networks with the target network. This feature is supposed to make the algorithm more stable compared to the standard version. Additionally, generating the targets using an older set of parameters (soft updates) adds a delay between the time an update to local networks is made and the time the update affects the target networks, making divergence or oscillations much more unlikely.

Since we are dealing with low dimensional feature vector observations, we utilize a technique called batch normalization on the state input. According to the paper, different components of the observation may have different physical units and the ranges may vary across environments. This can make it difficult for the network to learn effectively and make make it difficult to find hyperparameters which generalize across environments with different scales of state values. With batch normalization, we observed that learning happened more quickly.

Lastly, we tackle exploration, a major challenge in continouse action spaces by using a noise process (Ornstein-Uhlenbeck process), which reset in each episode. Additionally, we add a noise decay parameters so that the actions become less exploratory as time goes on.


### Hyperparameters

1. BUFFER_SIZE = int(1e6)

This is the size of the most recent memories we store during run time. In practice, we are not going to store all the experiences since beginning, but only retain a certain amount of most recent experiences. This model uses 100000 as buffer size, which contains roughly 100 most recent episodes.

2. BATCH_SIZE = 128

This is the number of experiences we randomly sample from the above pool for each learning update step

3. GAMMA = 0.99

This is the discounting factor use in calculate the present value of the action. 

4. TAU  = 1e-3

This is the proprotion of local networks we retain when copying the parameters of the local networks to target networks in every k time steps. In practice, we do not clone 100% of the local networks.

5. LR_ACTOR = 1e-4
   LC_CRITIC = 1e-4

This is the size of the update on local networks (actor and critic) parameters on each training step. Larger value may cause the model not improve or plateau too early after a certain number of episodes. Smaller value may take the model too long to get to the desired performance.

6. UPDATE_EVERY = 20

This is the number of time steps occuring between each learning step. As mentioned above, this serves as the delay between the syncing of local networks and target networks.

7. N_LEARN = 5

This is the number of consecutive training steps when we update the networks

8. NOISE_DECAY = 1e-6

This is the decay factor to reduce noise as the agent goes

### Model Architecture

#### Actor

Our model consiste of 2 fully connected hidden layers with input as the state representation (33 states) of the environment and output as the predicted action for individual actions (a vector size of 4).

Layer 1 has 300 units and Layer 2 has 200 units. The action space was clipped between -1 and 1.

#### Critic

Our model consiste of 2 fully connected hidden layers with input as the state representation (33 states) of the environment and output as a single predicted value for state-action pair (a vector size of 1). Actions input was introduced in the 2nd hidden layer.

Layer 1 has 300 units and Layer 2 has 204 (200+4) units.

### Future Ideas

There are 3 main ideas (coming from very recent researches) that the author would like to try to improve the agent performance:

-  Distributed Distributional Deterministic Policy Gradients (D4PG): we will modify the DDPG algorithm in 4 places. First, we will include a distributional critic update. In this enhancement, we'll introduce a random variable Z so that the update to critic will take a form of a distribution instead of a single target. Second, we will use distributed parallel actors, which is similar to the implementation of this project. Third, we'll utilize N-step returns in estimating the TD error. N is also generated randomly to create a distribution of expectations as part of the first modification. Lastly, we'll factor in prioritization of experience replay, where a priority p is assigned to each experience based on reward so that higher reward experiences can be sampled more often. (https://openreview.net/pdf?id=SyZipzbCb)

- Generalalized Advantage Estimation: the idea is to create a duelling Q-networks during the training. The architecture is similar to the version implemented in this project. However, the difference is this duelling network contains an advantage function that calculates the advantage that each actions would make. This value is then combined with the predicted state values to arrive at the final state-action values. (https://arxiv.org/abs/1511.06581)

- Trust Region Policy Otimization (TRPO): the idea is to use pixels of the picture as the input layer for the networks instead of the state representation. Because of dealing with pictures, our model architecture and pipeline have to adjusted to include image preprocessing steps and convolution layers instead of linear layers.

