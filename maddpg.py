# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as f
from utilities import soft_update, transpose_to_tensor, transpose_list
from buffer import ReplayBuffer
import numpy as np
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
UPDATE_EVERY = 2
NOISE = 6
NOISE_DECAY = 0.9999
NOISE_END = 0.0
NOISE_END_AFTER = 1e5
LR_ACTOR = 5e-4
LR_CRITIC = 5e-4
WEIGHT_DECAY = 0
DISCOUNT_FACTOR = 0.99
TAU = 0.06
N_TRAIN = 4

class MADDPG:
    def __init__(self, random_seed):
        super(MADDPG, self).__init__()

        self.seed = random_seed
        # actor input = local obs = 24
        # critic input = obs_full = 24+24 =48
        self.maddpg_agent = [DDPGAgent(self.seed, 24, 256, 128, 2, 48, 256, 128,LR_ACTOR,LR_CRITIC,WEIGHT_DECAY), 
                             DDPGAgent(self.seed, 24, 256, 128, 2, 48, 256, 128,LR_ACTOR,LR_CRITIC,WEIGHT_DECAY)]
        
        self.discount_factor = DISCOUNT_FACTOR

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE)

        self.tau = TAU
        self.Nstep = 0
        self.Noise = NOISE
        self.num_agents = 2

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents):
        """get actions from all agents in the MADDPG object"""
        obs_all_agents = [torch.tensor(i).float() for i in obs_all_agents]
        actions = [agent.act(obs, self.Noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        # print(actions)
        actions = torch.stack(actions).detach().numpy()
        actions = np.rollaxis(actions,1)
        # print(actions)
        return actions[0]

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, self.Noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents
        Centralized planning: training a centralized critic
        Decentralized execution: using individual actors 
        samples: each samples contains all observations for all N agents"""
        
        #Transpose of list of tensors for each components of samples: obs, action, reward, next obs and done
        obs, action, reward, next_obs, done = map(transpose_to_tensor,samples)

        #Concat list of tensors into a single tensor (used for centralized planning)
        obs_full = torch.cat(obs,1)
        next_obs_full = torch.cat(next_obs,1)
        
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        # target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full,target_actions)       
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        
        #Calculate Q estimate
        action = torch.cat(action, dim=1)
        # critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = agent.critic(obs_full,action)


        critic_loss = f.mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input (action predictions) to all agents (local only)
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs)]
        pred_actions = torch.cat(q_input, dim=1)
        # combine actor actions and full observations for actor input
        # q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(obs_full,pred_actions).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        # logger.add_scalars('agent%i/losses' % agent_number,
        #                    {'critic loss': cl,
        #                     'actor_loss': al},
        #                    self.iter)
        # print('/n agent%i/losses --- critic loss: %.5f ---- actor loss: %.2f ' % (agent_number,cl,al))

    def step(self, transitions):
        self.Nstep += 1
        #Adding experiences to buffer
        self.memory.push(transitions)
        #Sample experiences to learn
        if len(self.memory) > BATCH_SIZE and self.Nstep % UPDATE_EVERY == 0:
            #Update noise only once for each update round
            if self.Nstep <= NOISE_END_AFTER:
                self.Noise *= NOISE_DECAY
            else:
                self.Noise == NOISE_END
                # print(self.Noise)
            for i in range(N_TRAIN):
                for a_i in range(self.num_agents):                
                    samples = self.memory.sample(BATCH_SIZE)
                    self.update(samples, a_i)
                self.update_targets() #soft update the target network towards the actual network

    def update_targets(self):
        """soft update targets"""
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)

    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    def save_checkpoints(self, model_dir="./model_dir/"):
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor.state_dict(), "{0}actor_agent_{1}.pth".format(model_dir,i)) 
            torch.save(agent.critic.state_dict(), "{0}critic_agent_{1}.pth".format(model_dir,i))
    
    def load_checkpoints(self, model_dir="./model_dir/"):
         for i, agent in enumerate(self.maddpg_agent):
            agent.actor.load_state_dict(torch.load("{0}actor_agent_{1}.pth".format(model_dir,i)))
            agent.critic.load_state_dict(torch.load("{0}critic_agent_{1}.pth".format(model_dir,i)))
            
            




