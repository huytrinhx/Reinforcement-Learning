[//]: # (Image References)

[image1]: https://github.com/huytrinhx/DQN-Banana-Collector/blob/main/Images/ScoresChart.png

# Project 1: Navigation

### Overview

Using Deep Q Learning Neural Network, the agent showed it can solve the environment after more than 500 episodes. As mentioned in the ReadMe.md, the environment is considered solved if the agent's mean scores in the last 100 episodes is greater than 13. The chart below showed the agent score progression during the training

![Score Chart][image1]

### Algorithms

The full algorithm for training deep Q-networks is similar the algorithm explained in the DQN paper (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). For each time step in the episode, the agent selects and executes actions according to an epsilon-greedy policy based on Q-networks. Our epsilon decreases as we progress as we would like to algorithm to explore more random moves early before strictly relying on Q-networks to decide the next action. To determine the best action to take at each time step, the algorithm used a technique called experience replay. Experience replay means we store the agent's experiences at each time step for many episodes. Then, at the same time, we applied Q-learning updates which sampled the experiences randomly.

Another feature of the algorithm is to use a separate network for generating the target in the Q-learning update. Particularly, for every learn step (which happened once every k time steps), we clone the network Q to obtain the target network Q-hat, which was used to generate the target y. This feature is supposed to make the algorithm more stable compared to the standard version. Generating the targets using an older set of parameters adds a delay between the time an update to Q is made and the time the update affects the target y, making divergence or oscillations much more unlikely.

For detailed pseudo-code of deep Q-learning with experience replay, please refer to Algorithm 1 in the mentioned paper above.


### Hyperparameters

1. BUFFER_SIZE = int(1e5)

This is the size of the most recent memories we stored during run time. In practice, we are not going to store all the experiences since beginning, but only retain a certain amount of most recent experiences. This model uses 10000 as buffer size, which contains roughly 30 most recent episodes.

2. BATCH_SIZE = 64

This is the number of experiences we randomly sampled from the above pool for each learning update step

3. GAMMA = 0.95

This is the discounting factor used in calculate the present value of the action. 

4. TAU  = 1e-3

This is the size the update when we update the parameters of the target network in every k time steps. In practice, we do not clone 100% of the local Q-networks.

5. LR = 5e-4

This is the size of the update on local Q-networks parameters on each training step. Larger value may cause the model not improve or plateau too early after a certain number of episodes. Smaller value may take the model too long to get to the desired performance.

6. UPDATE_EVERY = 4

This is the number of time steps occuring between each learning step. As mentioned above, this serves as the delay between the update of local Q-networks and target Q-networks.

### Model Architecture

### Future Ideas

