# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
from buffer import ReplayBuffer
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
UPDATE_EVERY = 10
NOISE = 2
NOISE_DECAY = 0.9999

class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.01):
        super(MADDPG, self).__init__()

        # critic input = obs_full + actions = 24+2+2=30
        self.maddpg_agent = [DDPGAgent(24, 256, 156, 2, 52, 256, 156), 
                             DDPGAgent(24, 256, 156, 2, 52, 256, 156)]
        
        self.discount_factor = discount_factor

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE)

        self.tau = tau
        self.iter = 0
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
        self.Noise *= NOISE_DECAY
        actions = [agent.act(obs, self.Noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, self.Noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger):
        """update the critics and actors of all the agents """

        obs, action, reward, next_obs, done = map(transpose_to_tensor,samples)

        obs_full = torch.cat(obs,1)
        next_obs_full = torch.cat(next_obs,1)
        # print("/n Next Obs Full")
        # print(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        # print("/n Target Actions")
        # print(target_actions)
        target_actions = torch.cat(target_actions, dim=1)
        # print("/n Target Actions after torch cat")
        # print(target_actions)

        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        # print("Expected Reward generated by target critic")
        # print(y)
        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = agent.critic(critic_input)
        # print("Critic Agent Network")
        # print(q)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        # print("Critic Agent Loss")
        # print(critic_loss)
        # critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        # logger.add_scalars('agent%i/losses' % agent_number,
        #                    {'critic loss': cl,
        #                     'actor_loss': al},
        #                    self.iter)
        # print('/n agent%i/losses --- critic loss: %.5f ---- actor loss: %.2f ' % (agent_number,cl,al))

    def step(self, transitions, logger):
        self.Nstep += 1
        #Adding experiences to buffer
        self.memory.push(transitions)
        #Sample experiences to learn
        if len(self.memory) > BATCH_SIZE and self.Nstep % UPDATE_EVERY == 0:
            for a_i in range(self.num_agents):
                samples = self.memory.sample(BATCH_SIZE)
                self.update(samples, a_i, logger)


    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




