[//]: # (Image References)

[image1]: https://github.com/huytrinhx/DQN-Banana-Collector/blob/main/Images/ScoresChart.png

# Project 1: Navigation (Pixel-based)

### Overview

This is the pixel-based version of DQN where we used image instead of ray-based perception to train the Q-network. Keeping hyperparameters as they were for ray-based version and same model architecture as suggested by the original pixel-based DQN paper (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), I observe that training took longer and results was much more unstable. Also, because of the known issue with this version of Banana (https://github.com/Unity-Technologies/ml-agents/issues/1055), memories would accumulate as training goes on, I did not let it run for more than 300 episodes. However, this serves as a good foundation to adapt to other Unity environments of your own choice.

![Score Chart][image1]

### Algorithms

The full algorithm for training deep Q-networks is similar the algorithm explained in the DQN paper (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). For each time step in the episode, the agent selects and executes actions according to an epsilon-greedy policy based on Q-networks. Our epsilon decreases as we progress as we would like to algorithm to explore more random moves early before strictly relying on Q-networks to decide the next action. To determine the best action to take at each time step, the algorithm used a technique called experience replay. Experience replay means we store the agent's experiences at each time step for many episodes. Then, at the same time, we applied Q-learning updates which sampled the experiences randomly.

Another feature of the algorithm is to use a separate network for generating the target in the Q-learning update. Particularly, for every learn step (which happened once every k time steps), we clone the network Q to obtain the target network Q-hat, which was used to generate the target y. This feature is supposed to make the algorithm more stable compared to the standard version. Generating the targets using an older set of parameters adds a delay between the time an update to Q is made and the time the update affects the target y, making divergence or oscillations much more unlikely.

For detailed pseudo-code of deep Q-learning with experience replay, please refer to Algorithm 1 in the mentioned paper above.

### Image Preprocessing and Stacking Images

Both image preprocessing and stacking frames are important features mentioned in the original DQN Nature paper. Image preprocessing includes turning the picture into greyscales so that the dimension of the original picture as (1,84,84,3) is reduced to (1,84,84). As yellow banana and blue banana stills look differently in greyscale, this does not affect training accuracy yet improve learning efficiency.

As to stacking frames as an input to the network, I've found the following interpretation as a pretty solid answer to the original paper's ambiguity. (https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/). The stacking mechanism along with integration with replay buffer were executed nicely in this repo (https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-Keras/blob/master/Chapter%2011/DQN_Atari_v2.ipynb). As the result of stacking frames, the input to the convolution layers are 4 pictures of 84X84.


### Hyperparameters

1. BUFFER_SIZE = int(1e5)

This is the size of the most recent memories we stored during run time. In practice, we are not going to store all the experiences since beginning, but only retain a certain amount of most recent experiences. This model uses 10000 as buffer size, which contains roughly 30 most recent episodes.

2. BATCH_SIZE = 128

This is the number of experiences we randomly sampled from the above pool for each learning update step

3. GAMMA = 0.95

This is the discounting factor used in calculate the present value of the action. 

4. TAU  = 6e-2

This is the size the update when we update the parameters of the target network in every k time steps. In practice, we do not clone 100% of the local Q-networks.

5. LR = 5e-4

This is the size of the update on local Q-networks parameters on each training step. Larger value may cause the model not improve or plateau too early after a certain number of episodes. Smaller value may take the model too long to get to the desired performance.

6. UPDATE_EVERY = 10

This is the number of time steps occuring between each learning step. As mentioned above, this serves as the delay between the update of local Q-networks and target Q-networks.

7. STACK_SIZE = 4
Number of stacked frames together as input to Q-network.

### Model Architecture

Our model has the following layers:
- input layers has 4 channels representing stack of 4 images - dimension (1,4,84,84)
- 1 convolution layers - 32 filters of kernel size 8 - stride 4 - activated by relu function
- 1 convolution layers - 64 filters of kernel size 4 - stride 2 - activated by relu function
- 1 max pooling layer with kernel size = 4
- 1 convolution lerys - 64 filters of kernel size 1 - stride 1 - activated by relu function
- 1 max pooling layer with kernel size = 1
- 1 hidden fully connected layers with 512 nodes
- output layers with size of 4

### Future Ideas

There are few main ideas that the I would like to try as natural next steps:
- Experiement with other activation functions and model architecture
- Play with various hyperparameters
- Adapt this implementation to a newer version  of Banana Collector known as Food Collector  



