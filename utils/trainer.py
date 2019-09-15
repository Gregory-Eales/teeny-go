import logging
import time
import os

import torch
import numpy as np

from .multi_go_engine import MultiGoEngine

class GoTrainer(object):

    def __init__(self, network=None):

        if network==None:
            raise ValueError("no network supplied")

        # initialize model
        self.network = network.double()

        # save network attributes
        self.num_res = self.network.num_res_blocks
        self.num_channels = self.network.num_channels

        # set model name
        self.model_name = "Model-R{}-C{}".format(self.num_res, self.num_channels)

        # load game engine
        self.engine = MultiGoEngine()

    def save_model(self, version):
        path = "models/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filename = "Model-R{}-C{}-V{}.pt".format(self.num_res, self.num_channels, version)
        torch.save(self.network.state_dict(), path+filename)

    def load_model(self, version):
        path = "models/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filename = "Model-R{}-C{}-V{}.pt".format(self.num_res, self.num_channels, version)
        self.network.load_state_dict(torch.load(path+filename))

    def save_data(self):
        path = "data/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filenameX = "Model-R{}-C{}-V{}-DataX.pt".format(self.num_res, self.num_channels, version)
        filenameY = "Model-R{}-C{}-V{}-DataY.pt".format(self.num_res, self.num_channels, version)
    def load_data(self):
        pass

    def play_through_games(self, num_games):

        # reset and clear engine
        self.engine.reset_games(num_games)

        # main play loop
        while self.engine.is_playing_games():

            state_tensor = (torch.from_numpy(self.engine.get_active_game_states())).double()
            move_tensor = self.network.forward(state_tensor).detach().numpy()
            self.engine.take_game_step(move_tensor)


        # change game outcomes
        self.engine.finalize_game_data()

        return self.engine.get_all_data()

    def train_self_play(self, num_games=100, iterations=1):

        # assert inputs
        assert type(iterations)==int, "iterations must be an integer"
        assert type(num_games)==int, "number of games must be an integer"

        # loop through each iteration (index start at 1)
        for iter in range(1, iterations+1):

            # play through games
            x, y = self.play_through_games(num_games=num_games)

            x, y = (torch.from_numpy(x)).double(), (torch.from_numpy(y)).double()

            # train on new game data

            self.network.optimize(x, y, batch_size=x.shape[0], iterations=20)

            # save model

            self.save_model(version=iter)

            # save game data

            #self.save_data(iteration=iter)

    def train_data(self):
        pass
