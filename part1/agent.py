import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic  = torch.nn.Linear(self.hidden, 1)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0.0, 0.1) # upd
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        val = self.fc3_critic(x_critic).squeeze(-1)
        
        return normal_dist, val
    
    # define and rturn separate parameters for the two distinct optimization problems
    # used for the actor updates
    def actor_parameters(self):
        return (list(self.fc1_actor.parameters()) +
                list(self.fc2_actor.parameters()) +
                list(self.fc3_actor_mean.parameters()) +
                [self.sigma])

    # used for the critic updates
    def critic_parameters(self):
        return (list(self.fc1_critic.parameters()) +
                list(self.fc2_critic.parameters()) +
                list(self.fc3_critic.parameters()))


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        # separate optimizers, one for critic and one for the actor
        self.actor_optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=1e-4) # lr chosen w. hptuning
        self.critic_optimizer = torch.optim.Adam(self.policy.critic_parameters(), lr=3e-3) # lr chosen w. hptuning
        self.gamma = 0.999 # gamma chosen w. hptuning
        
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self, baseline, algorithm):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        if algorithm == 'REINFORCE':
            #
            # TASK 2:
            # compute discounted returns
            discounted_returns = discount_rewards(rewards, self.gamma)
            
            if baseline:
                discounted_returns -= 20.0
            
            # normalization of the returns 
            discounted_returns = (discounted_returns - discounted_returns.mean()) / (1e-8 + discounted_returns.std())
            
            # compute policy gradient loss function given actions and returns
            loss = -(action_log_probs * discounted_returns).mean()

            # upd here 
            
            # compute gradients and step the optimizer
            self.actor_optimizer.zero_grad()
            loss.backward()

            # future hp upd 
            # clipping of gradient 
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), max_norm=0.5)  
            self.actor_optimizer.step()
            return loss.item(), None
        
        else:
            # TASK 3:
            _, values = self.policy(states)
            with torch.no_grad():
                _, next_values = self.policy(next_states)
                
            # compute boostrapped discounted return estimates
            td_return_estimates = rewards + self.gamma * next_values * (1 - done)
            
            # compute advantage terms
            advantage_terms = td_return_estimates - values  

            advantage_terms = (advantage_terms.detach() - advantage_terms.detach().mean()) / \
                              (advantage_terms.detach().std() + 1e-8)
            
            # compute actor loss and critic loss
            actor_loss = -(action_log_probs * advantage_terms.detach()).mean()  
            critic_loss = F.mse_loss(values, td_return_estimates.detach())
            
            # compute gradients and step the optimizer
            # actor opt. 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor_parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            # critic opt.
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic_parameters(), max_norm=0.5)
            self.critic_optimizer.step()


            return actor_loss.item(), critic_loss.item()


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

