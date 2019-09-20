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

        # initialize elo agent rater
        self.rater = Rater()

        # initlize logger
        self.logger = logging.getLogger("Multi-Agent-Tester")

        # initilize something..

    # reset engine with n games
    def reset_engine(self, n=100):
        del(self.engine)
        self.engine = MultiGoEngine(num_games=n)

    def make_move(self, ai):
        state_tensor = (torch.from_numpy(self.engine.get_active_game_states())).cuda().type(torch.cuda.FloatTensor)
        move_tensor = self.network.forward(state_tensor)
        torch.cuda.empty_cache()
        self.engine.take_game_step(move_tensor.cpu().detach().numpy())

    # plays through n games a1 vs a2
    def play_throgh_games(self, a1, a2, num_games):

        #initilize go engine with n games
        self.reset_engine(num_games)

        # play through games

        while self.engine.is_playing_games():

            turn = self.engine.get_turn()

            if turn == 0:
                # get a1 moves
                self.make_move(a1)

            elif turn == 1:
                # get a2 moves
                self.make_move(a2)

            # if game is over
            elif turn == -4:
                break

    def get_win_rates(self):
            pass
