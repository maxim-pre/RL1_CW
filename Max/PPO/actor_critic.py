import torch 
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # policy mean head
        self.mu_head = nn.Linear(256, act_dim) 

        # value head
        self.v_head = nn.Linear(256, 1)

        # policy log std parameter - defines spread of action distribution
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):

        # add dimension if needed
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # extract shared features
        features = self.shared_layers(obs)

        # get policy mean
        mu = self.mu_head(features)

        # get value
        v = self.v_head(features).squeeze(-1)

        # get policy std
        std = torch.exp(self.log_std).expand_as(mu)

        return mu, std, v
    
    def distribution(self, obs):
        # helper function to get the normal distribution (guassian policy) given observations from current state
        mu, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        return dist, v
    
    def get_action(self, obs):
        # Used during rollout to collect experience 
        dist, value = self.distribution(obs)
        action = dist.sample()
        
        action_clipped = torch.clamp(action, -1.0, 1.0)  # clip action to be in [-1, 1]

        log_prob = dist.log_prob(action_clipped).sum(axis=-1) 

        if action_clipped.dim() == 2 and action_clipped.size(0) == 1:
            action_clipped = action_clipped.squeeze(0)

        return action_clipped.detach(), log_prob.detach(), value.detach()
    
    def evaluate_actions(self, obs, actions):
        # Used during PPO update to evaluate actions taken in the rollout

        dist, values = self.distribution(obs)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        return log_probs, entropy, values
    
    def get_value(self, obs):
        # helper function to get value given observations from current state
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            features = self.shared_layers(obs)
            v = self.v_head(features).squeeze(-1)
        
        return v.squeeze(0)
    

