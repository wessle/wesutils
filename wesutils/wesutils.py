import numpy as np
import torch
from collections import deque
import yaml
import pickle
import os
from datetime import datetime
from shutil import copyfile


# fully-connected networks

def single_layer_net(input_dim, output_dim,
                     hidden_layer_size=256,
                     activation='ReLU'):
    """
    Generate a fully-connected single-layer network for quick use.
    """

    activ = eval('torch.nn.' + activation)

    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_layer_size),
        active(),
        torch.nn.Linear(hidden_layer_size, output_dim))
    return net

def two_layer_net(input_dim, output_dim,
                  hidden_layer1_size=256,
                  hidden_layer2_size=256,
                  activation='ReLU'):
    """
    Generate a fully-connected two-layer network for quick use.
    """
    
    activ = eval('torch.nn.' + activation)

    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_layer1_size),
        activ(),
        torch.nn.Linear(hidden_layer1_size, hidden_layer2_size),
        activ(),
        torch.nn.Linear(hidden_layer2_size, output_dim)
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
