import logging

import torch
import numpy as np
import pyspiel
from tqdm import tqdm

from .agents.random_agent import RandomAgent
from .rater import Rater
from .multi_go_engine import MultiGoEngine



class MultiTester(object):

    def __init__(self):

        # initialize engine
        self.engine = MultiGoEngine()

    # reset engine with n games
    def reset_engine(self, n=100):
        del(self.engine)
        self.engine = MultiGoEngine(num_games=n)

    # plays through n games a1 vs a2
    def play_throgh_games(self, a1, a2, num_games):

        #initilize go engine with n games
        self.reset_engine(num_games)

        # play through games
        
