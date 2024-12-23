import os
import numpy as np
import logging
from train_test import train, test
import warnings
import json
from arg_parser import init_parser
#from setproctitle import setproctitle as ptitle
from enviroment import ARC_Env
import gymnasium as gym
from action_space import ARCActionSpace
import torch


if __name__ == "__main__":

    print('Running test in test.py')
    state_flat = torch.randn(64, 1800)
    shape_flat = torch.randn(64, 5)
    print('created state and shape tensors')
    x = torch.cat([state_flat, shape_flat], dim=-1)
    print(x.shape)
    print('-'*50)