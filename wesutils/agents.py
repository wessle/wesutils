import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import copy
import warnings
import numbers

import wesutils.utils as utils


# noise models for adding noise to DDPG agent

class NoiseModel:
    """General noise model prototype."""
    def __init__(self):
        pass
    
    def make_noise(self, state=None, time=None):
        pass


class GaussianNoise(NoiseModel):
    """Generate Gaussian noise."""
    
    def __init__(self, dim, mean=None, cov=None):
        
        NoiseModel.__init__(self)
        
        self.dim = dim
        
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.zeros(self.dim)
            
        if cov is not None:
            self.cov = cov
        else:
            self.cov = np.eye(self.dim)
        
    def __call__(self, state=None, time=None):
        return np.random.multivariate_normal(self.mean, self.cov)
        
        
class RLAgent:
    """General parent class for agents using specific algorithms."""

    def __init__(self, batch_size, action_dim, buff_len):
        
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.buffer = utils.Buffer(maxlen=buff_len)
    
    def sample_action(self, state):
        """Generate an action based on a given state. Save and return the
        action.
        """
        pass
    
    def update(self, reward, next_state):
        """Observe the reward and the next state, then carry out an update
        step.
        """
        pass


class DDPGAgent(RLAgent):
    """
    Agent that carries out the DDPG algorithm.

    The action noise is Gaussian by default.
    """
    
    def __init__(self, batch_size, action_dim, buff_len,
                 policy_network, critic_network, policy_lr, critic_lr,
                 gamma, tau,
                 action_noise_cov=0.1,
                 enable_cuda=True, optimizer=torch.optim.Adam,
                 grad_clip_radius=None, action_clip_radius=None):
        
        RLAgent.__init__(self, batch_size, action_dim, buff_len)
        
        # networks
        self.action_noise = GaussianNoise(action_dim,
                                          cov=action_noise_cov * np.eye(action_dim))
        self.pi = policy_network #.to(self.device)
        self.target_pi = copy.deepcopy(policy_network)
        self.Q = critic_network #.to(self.device)
        self.target_Q = copy.deepcopy(critic_network)
        
        self.__cuda_enabled = enable_cuda
        self.enable_cuda(self.__cuda_enabled, warn=False)
        # NOTE: self.device is defined when self.enable_cuda is called!
        
        # discount factor and tau
        self.gamma = gamma
        self.tau = tau

        
        # define optimizers
        self.pi_optim = optimizer(self.pi.parameters(), lr=policy_lr)
        self.Q_optim = optimizer(self.Q.parameters(), lr=critic_lr)
        
        # keep previous state and action
        self.state = None
        self.action = None

        # clip gradients and actions?
        self.grad_clip_radius = grad_clip_radius
        self.action_clip_radius = action_clip_radius
        
    @property
    def cuda_enabled(self):
        return self.__cuda_enabled
        
    def sample_action(self, state):
        """Get an action based on the current state."""
        self.state = state
        state = torch.FloatTensor(state).to(self.device)
        noise = torch.FloatTensor(self.action_noise(state)).to(self.device)
        action = self.pi(state) + noise
        self.action = action.cpu().detach().numpy()
        if self.action_clip_radius is not None:
            self.action = self.action.clip(-self.action_clip_radius,
                                           self.action_clip_radius)
        return self.action
        
    def _soft_parameter_update(self, params1, params2):
        for param1, param2 in zip(params1, params2):
            param1.data.copy_(
                    self.tau*param2.data + (1.0 - self.tau)*param1.data)

    def _save_rewards(self, filename):
        """Save accumulated rewards."""
        np.save(filename, self.rewards)
                               
    def update(self, reward, next_state, update_Q=True, update_pi=True):
        """Perform the DDPG update.
        
        If update_Q == update_pi == False, this step can be used to fill
        the buffer. If update_Q != update_pi == False, it can be used to
        perform policy evaluation (e.g. on a newly-loaded policy).
        """
        
        new_sample = (self.state, self.action, reward, next_state)
        self.buffer.append(new_sample)
        
        if update_Q and len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states = utils.arrays_to_tensors(
                    self.buffer.sample(self.batch_size), self.device)
            
            with torch.no_grad():
                target_actions = self.target_pi(next_states)
                target_Q_input = torch.cat(
                        [next_states, target_actions], dim=1)
                Q_targets = rewards.unsqueeze(1) + \
                    self.gamma * self.target_Q(target_Q_input)
                
            # minimize Q prediction error
            Q_input = torch.cat([states, actions], dim=1)
            Q_loss = F.mse_loss(self.Q(Q_input), Q_targets)
            self.Q_optim.zero_grad()
            Q_loss.backward()
            if self.grad_clip_radius is not None:
                torch.nn.utils.clip_grad_norm_(self.Q.parameters(),
                                               self.grad_clip_radius)
            self.Q_optim.step()
        
            if update_pi:
                # minimize expected cost as a function of policy
                pi_loss = self.Q(
                        torch.cat([states, self.pi(states)], dim=1)).mean()
                self.pi_optim.zero_grad()
                pi_loss.backward()
                if self.grad_clip_radius is not None:
                    torch.nn.utils.clip_grad_norm_(self.pi.parameters(),
                                                   self.grad_clip_radius)
                self.pi_optim.step()
                
            # update target parameters
            # TODO: figure out if it makes sense to update target_pi even
            # when pi is not updated
            self._soft_parameter_update(self.target_Q.parameters(),
                                        self.Q.parameters())
            self._soft_parameter_update(self.target_pi.parameters(),
                                        self.pi.parameters())


    def enable_cuda(self, enable_cuda=True, warn=True):
        """Enable or disable cuda and update models."""
        
        if warn:
            warnings.warn("Converting models between 'cpu' and 'cuda' after "
                          "initializing optimizers can give errors when using "
                          "optimizers other than SGD or Adam!")
        
        self.__cuda_enabled = enable_cuda
        self.device = torch.device(
                'cuda' if torch.cuda.is_available() and self.__cuda_enabled \
                else 'cpu')
        self.pi.to(self.device)
        self.target_pi.to(self.device)
        self.Q.to(self.device)
        self.target_Q.to(self.device)
        
    def load_models(self, filename, enable_cuda=True, continue_training=True):
        """Load policy and value functions. Copy them to target functions."""
        
        models = torch.load(filename)

        self.pi.load_state_dict(models['pi_state_dict'])
        self.target_pi = copy.deepcopy(self.pi)
        self.Q.load_state_dict(models['Q_state_dict'])
        self.target_Q = copy.deepcopy(self.Q)
        
        if continue_training:
            self.pi.train()
            self.target_pi.train()
        else:
            self.pi.eval()
            self.target_pi.eval()
            
        self.enable_cuda(enable_cuda, warn=False)
        
    def save_checkpoint(self, filename):
        """Save state_dicts of models and optimizers."""
        
        torch.save({
                'using_cuda': self.__cuda_enabled,
                'pi_state_dict': self.pi.state_dict(),
                'target_pi_state_dict': self.target_pi.state_dict(),
                'Q_state_dict': self.Q.state_dict(),
                'target_Q_state_dict': self.target_Q.state_dict(),
                'pi_optimizer_state_dict': self.pi_optim.state_dict(),
                'Q_optimizer_state_dict': self.Q_optim.state_dict(),
        }, filename)
    
    def load_checkpoint(self, filename, continue_training=True):
        """Load state_dicts for models and optimizers."""
        
        checkpoint = torch.load(filename)
        
        self.pi.load_state_dict(checkpoint['pi_state_dict'])
        self.target_pi.load_state_dict(checkpoint['target_pi_state_dict'])
        self.Q.load_state_dict(checkpoint['Q_state_dict'])
        self.target_Q.load_state_dict(checkpoint['target_Q_state_dict'])
        self.pi_optim.load_state_dict(checkpoint['pi_optimizer_state_dict'])
        self.Q_optim.load_state_dict(checkpoint['Q_optimizer_state_dict'])
        
        if continue_training:
            self.pi.train()
            self.target_pi.train()
            self.Q.train()
            self.target_Q.train()
            
        else:
            self.pi.eval()
            self.target_pi.eval()
            self.Q.eval()
            self.target_Q.eval()
        
        self.enable_cuda(checkpoint['using_cuda'], warn=False)




# end
