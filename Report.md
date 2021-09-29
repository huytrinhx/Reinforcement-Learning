[//]: # (Image References)

[image1]: https://github.com/huytrinhx/MADPPG-Play-Tennis/blob/main/ScoreChart.JPG

# Project 3: MADDPG for Multiagent Environment: Collaboration vs Competition

### Overview

Adapting Deep Deterministic Policy Gradient (DDPG) Network to multi-agent environment, the agent showed it can solve the environment in less than 2000 episodes. As mentioned in the ReadMe.md, the environment is considered solved if the agents' mean scores in the last 100 episodes is greater than 0.5. The chart below showed the agent score progression during the training

![Score Chart][image1]

### Algorithms

The full algorithm for training DDPG is fully explained in section Algorithm of the DDPG paper (https://arxiv.org/abs/1509.02971). You can also view my interpretation of the algorithm in this repo (https://github.com/huytrinhx/DDPG-Robotic-Arm/main/Report.md).

In order to apply DDPG to multi-agent environment, I make two following important modifications to the original DDPG:

- Centralized Training: The critic is updated with the shared experience replay during the training step. They take in all (global) observations of the environment and actions of all agents in the environment.

- Decentralized Execution: Each agent's actor is responsible for producing the action based on the agent's local observation of the environment.

The full algorithm for MADDPG is fully explained in section Algorithm of the MADDPG paper (https://arxiv.org/abs/1706.02275)

### Hyperparameters

1. BUFFER_SIZE = int(1e6)

This is the size of the most recent memories we store during run time. In practice, we are not going to store all the experiences since beginning, but only retain a certain amount of most recent experiences. This model uses 100000 as buffer size, which contains roughly 300 most recent episodes.

2. BATCH_SIZE = 256

This is the number of experiences we randomly sample from the above pool for each learning update step

3. GAMMA = 0.99

This is the discounting factor use in calculate the present value of the action. 

4. TAU  = 6e-2

This is the proprotion of local networks we retain when copying the parameters of the local networks to target networks in every k time steps. In practice, we do not clone 100% of the local networks.

5. LR_ACTOR = 5e-4
   LC_CRITIC = 5e-4

This is the size of the update on local networks (actor and critic) parameters on each training step. Larger value may cause the model not improve or plateau too early after a certain number of episodes. Smaller value may take the model too long to get to the desired performance.

6. UPDATE_EVERY = 2

This is the number of time steps occuring between each learning step. As mentioned above, this serves as the delay between the syncing of local networks and target networks.

7. N_TRAIN = 4

This is the number of consecutive training steps when we update the networks

8. NOISE = 6 
   NOISE_DECAY = 0.9999
   NOISE_END = 0.0
   NOISE_END_AFTER = 10000

In this project, I deloy a technique called Exploratory Boost. In the beginning, noise scale start at 6 (NOISE) and then decay at a rate of 0.9999 (NOISE_DECAY) in each update. After timestep 10000 (NOISE_END_AFTER), it will set noise to 0 (NOISE_END), when the actions of actor are determined deterministically by actor network.

### Model Architecture

In this project, 2 agents interact with each other and they have the same model architecture.

#### Actor (x2)

Each actor consists of 2 fully connected hidden layers with input as the state representation (24 states) of the environment and output as the predicted action for individual actions (a vector size of 2).

Layer 1 has 256 units and Layer 2 has 128 units. The action output space was clipped between -1 and 1.

#### Critic (x2)

Each critic consists of 2 fully connected hidden layers with input as full state representation of 2 agents (24 * 2 states) of the environment and output as a single predicted value for state-action pair (a vector size of 1). Actions input (2 * 2 vector size, full action representation of 2 agents) was introduced in the 2nd hidden layer.

Layer 1 has 256 units and Layer 2 has 132 (128+4) units.

### Future Ideas

There are 3 main ideas (coming from recent researches) that the author would like to try to improve the agent performance:

-  Twin Delayed DDPG: 
-  Prioritized Experience Replay:
-  Pixel-based Training: 

### On reproducibility




