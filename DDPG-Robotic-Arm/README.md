<<<<<<< HEAD
# Reinforcement-Learning
 A collection of projects under Deep Reinforcement Learning 
=======
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project Name: DDPG for Continuous Actions Environment (Project 2 for Deep Reinforcement Learning Nanodegree)


### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. An episode is considered done after 1000 time steps.

The task is episodic, and in order to solve the environment, our agent must get an average score of +30 over 100 consecutive episodes.

### Distributed Training

For this project, we will have two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.

### Solving the Environment

Note that this project submission only solve the 2nd of the two versions of the environment. The first version used the exact hyperparameters and model architecture as the second version. The point is to demonstrate the big difference in multi-agent learning

#### The First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### The Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Getting Started (Alternative for 64-bit Windows users)

Alternatively, you can clone the entire repo which has already loaded the Reacher.exe environment. However, please make sure you are using a 64-bit Windows machine otherwise the environment won't work.

### Downloading and Installing Dependencies
In addition to downloading the Reacher environment above, please make sure you download and install the following dependences for your python environment (ex.. anaconda...)

tensorflow==1.7.1
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==0.4.0
pandas
scipy
ipykernel 

Alternatively, you can follow the following instructions from the Dependencies section of the Deep Reinforcement Learning nanodegree repo (https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md) to complete the set up the python environment as well as all necessary libraries.

### Files in Repo

- Continuous_Control_Single_Agent.ipynb: the notebook that walks you through the process of starting the environment, training the single agent.
- Continuous_Control_Multi_Agents.ipynb: slightly modify the pointer to Multi-agent Reacher environment and the learning step during training to adapt to 20-agent environment
- ddpg_agent.py: an object that contains all methods for our DDPG agent and hyperparameters.
- model.py: an object that defines our model architecture.
- checkpoint_actor.pth: the saved model weights of our actor (policy) model after training.
- checkpoint_critic.pth: the saved model weights of our critic (value) model after training.
- Report.md: further explanation of model, hyperparameters and future ideas.

### Instructions

Follow the instructions in `Continuous_Control_Single_Agent.ipynb` to get started with training your own agent. Then follow the instructions in `Continuous_Control_Multi_Agents.ipynb` to train multiple agents at once.
>>>>>>> ddpg-robotic-arm/main
