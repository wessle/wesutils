import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
import torch.nn as nn
from collections import deque
import yaml
import pickle
import os
from datetime import datetime
from shutil import copyfile
import random
import numbers
import functools
import operator


# fully-connected networks

def single_layer_net(input_dim, output_dim,
                     hidden_layer_size=256,
                     activation='ReLU'):
    """
    Generate a fully-connected single-layer network for quick use.
    """

    activ = eval('nn.' + activation)

    net = nn.Sequential(
        nn.Linear(input_dim, hidden_layer_size),
        activ(),
        nn.Linear(hidden_layer_size, output_dim))
    return net

def two_layer_net(input_dim, output_dim,
                  hidden_layer1_size=256,
                  hidden_layer2_size=256,
                  activation='ReLU'):
    """
    Generate a fully-connected two-layer network for quick use.
    """
    
    activ = eval('nn.' + activation)

    net = nn.Sequential(
        nn.Linear(input_dim, hidden_layer1_size),
        activ(),
        nn.Linear(hidden_layer1_size, hidden_layer2_size),
        activ(),
        nn.Linear(hidden_layer2_size, output_dim)
    )
    return net


# torch tensor manipulation functions

def array_to_tensor(array, device):
    """Convert numpy array to tensor."""

    return torch.FloatTensor(array).to(device)

def arrays_to_tensors(arrays, device):
    """Convert iterable of numpy arrays to tuple of tensors."""
    # TODO: use *args in place of arrays
    
    return tuple([array_to_tensor(array, device) for array in arrays])

def copy_parameters(model1, model2):
    """Overwrite model1's parameters with model2's."""
    
    model1.load_state_dict(model2.state_dict())


# config file manipulation and logging functions

def load_config(filename):
    """Load and return a config file."""

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_logdir(directory, algorithm, env_name, config_path):
    """
    Create a directory inside the specified directory for
    logging experiment results and return its path name.

    Include the environment name (Sharpe, etc.) in the directory name,
    and also copy the config
    """

    experiment_dir = f"{directory}/{algorithm}-{env_name}-{datetime.now():%Y-%m-%d_%H:%M:%S}"
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if config_path is not None:
        copyfile(config_path, f"{experiment_dir}/config.yml")

    return experiment_dir

def save_object(obj, filename):
    """Save an object, e.g. a list of returns for an experiment."""
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_object(filename):
    """Load an object, e.g. a list of returns to be plotted."""
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


class Buffer:
    """
    Circular experience replay buffer for RL environments built around
    collections.deque.

    The 'maxlen' keyword argument must be passed. An optional iterable
    can also be passed to initialize the buffer.
    """
    
    def __init__(self, *args, maxlen=None):

        assert len(args) <= 1, 'Pass either no iterable or 1 iterable'
        assert maxlen is not None, '"maxlen" must be specified'

        deque_input = [] if len(args) == 0 else args[0]
        self.__buffer = deque(deque_input, maxlen)

    @property
    def buffer(self):
        return self.__buffer

    def __eq__(self, other):
        return (len(self) == len(other)) and \
                all(x == y for x, y in zip(self, other))

    def __hash__(self):
        hashes = (hash(x) for x in self)
        return functools.reduce(operator.xor, hashes, 0)

    def __getitem__(self, index):
        """Return self[index]."""

        if isinstance(index, numbers.Integral):
            return self.__buffer[index]
        else:
            msg = '{.__name__} indices must be integers'
            raise TypeError(msg.format(type(self)))

    def __iter__(self):
        """Return iter(self)."""

        return iter(self.__buffer)

    def __len__(self):
        """Return length of buffer."""

        return len(self.__buffer)

    def __repr__(self):
        """Return repr(self)."""
        index = repr(self.__buffer).find('[')
        return 'Buffer(' + repr(self.__buffer)[index:]
        
    def sample(self, batch_size):
        """Randomly sample a batch."""

        batch = random.sample(self.__buffer, batch_size)
        return tuple(list(x) for x in zip(*batch))
    
    def append(self, sample):
        """Append a sample."""

        self.__buffer.append(sample)


class TorchBuffer:
    """
    Buffer for storing tuples of flat torch tensors as a 2D torch tensor.

    Note that sampling occurs with replacement.
    """

    def __init__(self, maxlen=100000):

        self.maxlen = maxlen
        self.currlen = 0
        self.currindex = 0
        self.tensorlens = None
        self.__buffer = None

    @property
    def buffer(self):
        return self.__buffer

    def __len__(self):
        return self.currlen

    def append(self, tensor_tuple):
        """Add a new entry to the buffer."""

        assert all(torch.is_tensor(elem) for elem in tensor_tuple), \
                'All elements of the input tuple must be torch tensors'

        tensor_tuple = tuple(torch.flatten(elem) for elem in tensor_tuple)

        if self.__buffer is None:
            self.tensorlens = [len(elem) for elem in tensor_tuple]
            self.__buffer = torch.zeros(self.maxlen, sum(self.tensorlens))
        self.__buffer[self.currindex] = torch.cat(tensor_tuple, 0)

        self.currlen = min(self.currlen + 1, self.maxlen)
        self.currindex = (self.currindex + 1) % self.maxlen

    def sample(self, batch_size):
        """Randomly sample a batch."""

        indices = torch.randint(0, self.currlen, size=(batch_size,))
        return torch.split(self.__buffer[indices], self.tensorlens, 1)


# policy networks

class PolicyNetwork(nn.Module):
    """Base class for stochastic policy networks."""

    def __init__(self):
        super().__init__()

    def forward(self, state):
        """Take state as input, then output the parameters of the policy."""

        raise NotImplemented("forward not implemented.")

    def sample(self, state):
        """
        Sample an action based on the model parameters given the current state.
        """

        raise NotImplemented("sample not implemented.")


class GaussianPolicyNetworkBase(PolicyNetwork):
    """
    Base class for Gaussian policies.

    Desired two-headed network outputting mean and covariance needs to be
    implemented.
    """

    def __init__(self, state_dim, action_dim,
                 log_std_min=-20, log_std_max=5):

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.device = torch.device('cpu')

    def set_device(self, device):
        """Set the device."""

        self.device = device
        self.to(self.device)

    def sample(self, state, no_log_prob=False):
        """
        Sample from the Gaussian distribution with mean and covariance
        output by the two-headed policy network evaluated on the current state.
        """

        mean, cov = self.forward(state)
        dist = td.multivariate_normal.MultivariateNormal(
            mean, torch.eye(self.action_dim).to(self.device) * cov)
        action = dist.rsample()
        
        return_val = (action, dist.log_prob(action)) if not no_log_prob else action
        return return_val


class GaussianPolicyTwoLayer(GaussianPolicyNetworkBase):
    """
    Simple two-layer, two-headed Gaussian policy network.

    If simple_cov == True, the covariance matrix always takes the form
        
        sigma * I,

    where sigma is a scalar and I is an identity matrix with appropriate
    dimensions.
    """

    def __init__(self, state_dim, action_dim,
                 simple_cov=True,
                 hidden_layer1_size=256,
                 hidden_layer2_size=256,
                 activation='relu',
                 log_std_min=-20, log_std_max=3,
                 weight_init_std=0.0001):

        super().__init__(state_dim, action_dim,
                         log_std_min, log_std_max)

        self.simple_cov = simple_cov
        self.activation = eval('F.' + activation) # same activation everywhere
        
        # set the output dimension of the log_std network
        cov_output_dim = 1 if self.simple_cov else self.action_dim

        # define the network layers
        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.mean = nn.Linear(hidden_layer2_size, self.action_dim)
        self.log_std = nn.Linear(hidden_layer2_size, cov_output_dim)

        # initialize the weights of the layers
        nn.init.normal_(self.linear1.weight, std=weight_init_std)
        nn.init.normal_(self.linear1.bias, std=weight_init_std)
        nn.init.normal_(self.linear2.weight, std=weight_init_std)
        nn.init.normal_(self.linear2.bias, std=weight_init_std)
        nn.init.normal_(self.mean.weight, std=weight_init_std)
        nn.init.normal_(self.mean.bias, std=weight_init_std)
        nn.init.normal_(self.log_std.weight, std=weight_init_std)
        nn.init.normal_(self.log_std.bias, std=weight_init_std)
        
    def forward(self, state):
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        mean = self.mean(x)
        cov = torch.clamp(self.log_std(x),
                          self.log_std_min, self.log_std_max).exp()
        cov = cov.unsqueeze(dim=2) * torch.eye(
            self.action_dim).to(self.device)

        return mean, cov






# end
