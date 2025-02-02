#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import logging


def to_numpy(var, device=None):
    """
    Convert a PyTorch tensor to a NumPy array, handling the device.

    Args:
        var (torch.Tensor): The PyTorch tensor to convert.
        device (torch.device or None): The device type (e.g., 'cuda', 'mps', 'cpu').

    Returns:
        np.ndarray: The NumPy array.
    """
    if not isinstance(var, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")

    # Move tensor to the specified device if provided
    if device is not None:
        var = var.to(device)

    # Ensure the tensor is on CPU before converting to NumPy
    if var.device.type != 'cpu':
        var = var.cpu()

    # Detach from the computational graph and convert to NumPy
    return var.detach().numpy()

def to_tensor(ndarray, requires_grad=False, device=None):
    """
    Convert a NumPy array to a PyTorch tensor with the specified gradient and device settings.
    """
    tensor = torch.from_numpy(ndarray).float()
    if device:
        tensor = tensor.to(device)
    tensor.requires_grad = requires_grad
    return tensor

def soft_update(target, source, tau_update: float):
    """
    Performs a soft update of the target network parameters.
    
    target = (1 - tau) * target + tau * source
    
    Args:
        target (torch.nn.Module): Target network.
        source (torch.nn.Module): Source network (usually the latest model).
        tau_update (float): Interpolation parameter (0 < tau < 1).
    """
    with torch.no_grad():  # Disable gradient tracking for efficiency
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.copy_(target_param * (1.0 - tau_update) + source_param * tau_update)

def hard_update(target, source):
    """
    Copies all parameters from source network to target network (hard update).
    
    Args:
        target (torch.nn.Module): Target network to be updated.
        source (torch.nn.Module): Source network to copy parameters from.
    """
    with torch.no_grad():  # Disable gradient tracking for efficiency
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)  # Direct copy without unnecessary NumPy conversion


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device