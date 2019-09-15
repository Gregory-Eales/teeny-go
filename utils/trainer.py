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
        self.network = network

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

    def save_data(self, x, y, version):
        path = "data/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filenameX = "Model-R{}-C{}-V{}-DataX.pt".format(self.num_res, self.num_channels, version)
        filenameY = "Model-R{}-C{}-V{}-DataY.pt".format(self.num_res, self.num_channels, version)
        torch.save(x, path+filenameX)
        torch.save(y, path+filenameY)

    def load_data(self):
        pass

    def play_through_games(self, num_games, is_cuda=False):

        # reset and clear engine
        self.engine.reset_games(num_games)

        if is_cuda:

            self.network.cuda()
            # main play loop
            while self.engine.is_playing_games():

                state_tensor = (torch.from_numpy(self.engine.get_active_game_states())).cuda().type(torch.cuda.FloatTensor)

                move_tensor = self.network.forward(state_tensor)
                torch.cuda.empty_cache()
                self.engine.take_game_step(move_tensor.cpu().detach().numpy())

        else:

            # main play loop
            while self.engine.is_playing_games():

                state_tensor = (torch.from_numpy(self.engine.get_active_game_states()))
                move_tensor = self.network.forward(state_tensor).numpy()
                self.engine.take_game_step(move_tensor)

        # change game outcomes
        self.engine.finalize_game_data()

        return self.engine.get_all_data()

    def train_self_play(self, num_games=100, iterations=1, is_cuda=False):

        # assert inputs
        assert type(iterations)==int, "iterations must be an integer"
        assert type(num_games)==int, "number of games must be an integer"

        # loop through each iteration (index start at 1)
        for iter in range(1, iterations+1):

            # play through games
            x, y = self.play_through_games(num_games=num_games, is_cuda=is_cuda)

            x, y = (torch.from_numpy(x)), (torch.from_numpy(y))

            print(x.shape)

            if is_cuda:
                x = x.cuda().type(torch.cuda.FloatTensor)
                y = y.cuda().type(torch.cuda.FloatTensor)
            # train on new game data
            self.network.optimize(x, y, batch_size=10000, iterations=100, alpha=0.001)

            # save model
            self.save_model(version=iter)

            # save game data
            self.save_data(x, y, version=iter)

            del(x)
            del(y)

            print("Logger: finished iteration", iter)

    def train_data(self):
        pass
