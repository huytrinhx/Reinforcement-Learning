[//]: # (Image References)

[image1]: https://github.com/huytrinhx/Reinforcement-Learning/blob/main/DQN-Banana-Collector/Perception-Ray-based/Images/ScoresChart.png

# Project 1: Navigation

### Overview

Using Deep Q Learning Neural Network, the agent showed it can solve the environment after more than 500 episodes. As mentioned in the ReadMe.md, the environment is considered solved if the agent's mean scores in the last 100 episodes is greater than 13. The chart below showed the agent score progression during the training. 

![Score Chart][image1]

### Algorithms (Baseline)

The full algorithm for training deep Q-networks is similar the algorithm explained in the DQN paper (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). For each time step in the episode, the agent selects and executes actions according to an epsilon-greedy policy based on Q-networks. Our epsilon decreases as we progress as we would like to algorithm to explore more random moves early before strictly relying on Q-networks to decide the next action. To determine the best action to take at each time step, the algorithm used a technique called experience replay. Experience replay means we store the agent's experiences at each time step for many episodes. Then, at the same time, we applied Q-learning updates which sampled the experiences randomly. 

Another feature of the algorithm is to use a separate network for generating the target in the Q-learning update. Particularly, for every learn step (which happened once every k time steps), we clone the network Q to obtain the target network Q-hat, which was used to generate the target y. This feature is supposed to make the algorithm more stable compared to the standard version. Generating the targets using an older set of parameters adds a delay between the time an update to Q is made and the time the update affects the target y, making divergence or oscillations much more unlikely.

For detailed pseudo-code of deep Q-learning with experience replay, please refer to Algorithm 1 in the mentioned paper above.

### Prioritized Experience Replay


We make a modification in the mechanism which an experience is sampled by attaching a weight to each experience in the replay buffer. The intuition is that not all experience is equal thus if we can replay important experiences more frequently, we can achieve more efficient learning. (https://arxiv.org/pdf/1511.05952.pdf)

In particular, I chose to implement the TD-error prioritization, in which the computed TD-error in each experience represents the sampling probability weight of that experience. To avoid greedy prioritization that might exarcerbate the lack of diversity, we implemented a noise constant (1e-5 or 1e-6) to ensure non-zero probability even for lowest-priority transition. This variant, according to the paper, is called propotional prioritization.

To efficiently sample from replay buffer, the complexity cannot depend on buffer size. Instead, we're using sum-tree data structures where a parent node is the sum of its children. The leaf nodes represent each individual experience's priority. This provides a efficient way of calculating the cumulative sum of priorities, with time complexity (O log N) updates and sampling. To sample a minibatch of size k, the range [0, pTotal] is divided equally into k ranges. Next, a value is uniformly sampled from each range range. Finally the transitions that correspond to each of these sampled values are retrieved from the tree. 


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

Our model consisted of 2 fully connected hidden layers with input as the state representation (37 states) of the environment and output as the predicted Q-values for individual actions (4 actions). Each fully connected hidden layers had 64 units.

### Future Ideas

There are main ideas that I would like to try as natural next steps:

- Prioritized Experience Replay: Unfortunately, I did not see significant improvement in agent'performance in this environment based on result of limited tries. However, I supposed this implementation may be helpful for other environments. Also, more seeds should be used to confirm the range of prioritized agents' performance compared to vanilla agent. In this regards, running from command-line on cloud resources with logging will make running experiments more efficient rather than on notebook and local computer.

- Duelling DQN: the idea is to create a duelling Q-networks during the training. The architecture is similar to the version implemented in this project. However, the difference is this duelling network contains an advantage function that calculates the advantage that each actions would make. This value is then combined with the predicted state values to arrive at the final state-action values. (https://arxiv.org/abs/1511.06581)


