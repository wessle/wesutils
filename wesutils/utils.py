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


# neural networks

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


class DQN(nn.Module):
    """
    Convolutional neural network for use as a deep Q network in a deep
    Q-learning algorithm.

    Network contains two convolutional layers followed by two fully
    connected layers with ReLU activation functions.
    """

    def __init__(self, in_channels, num_actions,
                 example_state_tensor,  # example state converted to torch tensor
                 out_channels1=16, kernel_size1=3, stride1=2,
                 in_channels2=16, out_channels2=32, kernel_size2=2, stride2=1,
                 out_features3=256):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size1, stride1)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size2, stride2)

        in_features3 = self.conv2(
            self.conv1(example_state_tensor)).view(-1).shape[0]

        self.fc3 = nn.Linear(in_features3, out_features3, bias=True)
        self.head = nn.Linear(out_features3, num_actions, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))  # flattens Conv2d output
        return self.head(x)


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
    Buffer for easily storing and sampling torch tensors of experiences
    for batch training of RL algorithms.
    """

    Experience = namedtuple('Experience', ('state', 'action', 'reward',
                                           'next_state', 'done'))

    def __init__(self, maxlen=100000):
        self.__buffer = deque(maxlen=maxlen)

    @property
    def buffer(self):
        return self.__buffer

    def __len__(self):
        return len(self.__buffer)

    def append(self, state, action, reward, next_state, done):
        """
        Append to the buffer.

        NOTE: state, action, reward, next_state, and done should all be
        torch tensors of an appropriate kind. The burden is on the user
        to ensure this, as no checks are performed.
        """

        self.__buffer.append(
            self.Experience(state, action, reward, next_state, done)
        )

    def sample(self, batch_size, device=torch.device('cpu')):
        """
        Sample a batch of experiences:
            states, actions, rewards, next_states, dones
        and return them in nice torch.tensor form.

        TODO: be more concise.
        """

        batch_size = min(batch_size, len(self))
        indices = np.random.randint(0, len(self), size=(batch_size,))
        sample = [self.__buffer[i] for i in indices]
        states = torch.cat([elem.state for elem in sample]).to(device)
        actions = torch.cat([elem.action for elem in sample]).to(device)
        rewards = torch.cat([elem.reward for elem in sample]).to(device)
        next_states = torch.cat([elem.next_state for elem in sample]).to(device)
        dones = torch.cat([elem.done for elem in sample]).to(device)

        return states, actions, rewards, next_states, dones


class HeapBuffer:
    """
    Fixed-length buffer using a min-heap to keep only the elements with
    largest keys. The buffer is configured to store and return 0- and 1D
    torch tensors by default.

    Note that the heap is (partially) ordered by the value of the first
    element in each entry.

    As of Python 3, a counter element has to be appended as the second
    element of every sample, because heappush breaks when the second
    elements can't be compared, e.g. when using torch.Tensors of different
    sizes.
    """

    def __init__(self, buff_len, is_torch_buffer=True):

        self.buff_len = buff_len
        self.__heap = []
        self.__is_torch_buffer = is_torch_buffer
        self.counter = itertools.count()

    @property
    def buffer(self):
        return self.__heap

    @property
    def is_torch_buffer(self):
        return self.__is_torch_buffer

    def __len__(self):
        return len(self.__heap)

    def __repr__(self):
        """Return repr(self)."""

        index = repr(self.__heap).find('[')
        return 'HeapBuffer(' + repr(self.__heap)[index:] + ')'

    def sample(self, batch_size):
        """Randomly sample a batch."""

        batch = random.sample(self.__heap, batch_size)
        batch = enumerate(zip(*batch))
        return tuple(list(elem) for i, elem in batch if i != 1) \
                if not self.is_torch_buffer \
                else tuple(torch.stack(elem, dim=0) for i, elem in batch \
                           if i != 1)

    def append(self, sample):
        """
        Append a sample.

        Make sure to maintain the size of the heap and the heap invariant
        once the heap has maximum length.
        """

        sample = (sample[0], next(self.counter), *sample[1:])

        if len(self) < self.buff_len:
            heapq.heappush(self.__heap, sample)
        else:
            heapq.heapreplace(self.__heap, sample)


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
        if len(state.shape) < 2:
            state = state.reshape(1, 1)
        x = self.activation(self.linear1(state))
        x = self.activation(self.linear2(x))
        mean = self.mean(x)
        cov = torch.clamp(self.log_std(x),
                          self.log_std_min, self.log_std_max).exp()
        cov = cov.unsqueeze(dim=2) * torch.eye(
            self.action_dim).to(self.device)

        return mean, cov






# end
