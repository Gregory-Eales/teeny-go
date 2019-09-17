import logging
import time
import os

import torch
import numpy as np

from .multi_go_engine import MultiGoEngine
from .tester import Tester

class Trainer(object):

    def __init__(self, network=None):

        # make sure network is inputed
        if network==None:
            raise ValueError("no network supplied")

        # initialize model
        self.network = network

        # initialize tester
        self.tester = Tester()

        # load game engine
        self.engine = MultiGoEngine()

        # initialize logger
        self.logger = logging.getLogger(name="Trainer")

        # save network attributes
        self.num_res = self.network.num_res_blocks
        self.num_channels = self.network.num_channels

        # set model name
        self.model_name = "Model-R{}-C{}".format(self.num_res, self.num_channels)

    # saves model to models file
    def save_model(self, version):
        path = "models/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filename = "Model-R{}-C{}-V{}.pt".format(self.num_res, self.num_channels, version)
        torch.save(self.network.state_dict(), path+filename)

    # loads model from models file
    def load_model(self, version):
        path = "models/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filename = "Model-R{}-C{}-V{}.pt".format(self.num_res, self.num_channels, version)
        self.network.load_state_dict(torch.load(path+filename))

    # saves data to data file
    def save_data(self, x, y, version):
        path = "data/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filenameX = "Model-R{}-C{}-V{}-DataX.pt".format(self.num_res, self.num_channels, version)
        filenameY = "Model-R{}-C{}-V{}-DataY.pt".format(self.num_res, self.num_channels, version)
        torch.save(x, path+filenameX)
        torch.save(y, path+filenameY)

    # loads data from data file
    def load_data(self):
        pass

    # plays through n games, returns game data
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

                state_tensor = (torch.from_numpy(self.engine.get_active_game_states())).float()
                move_tensor = self.network.forward(state_tensor).detach().numpy()
                self.engine.take_game_step(move_tensor)

        # change game outcomes
        self.engine.finalize_game_data()

        # return game data tensors
        return self.engine.get_all_data()

    def train_self_play(self, num_games=100, iterations=1, is_cuda=False):

        # assert inputs
        assert type(iterations)==int, "iterations must be an integer"
        assert type(num_games)==int, "number of games must be an integer"

        # loop through each iteration (index start at 1)
        for iter in range(1, iterations+1):

            # play through games
            x, y = self.play_through_games(num_games=num_games, is_cuda=is_cuda)

            # convert to torch tensors
            x, y = (torch.from_numpy(x)), (torch.from_numpy(y))

            print(x.shape)

            if is_cuda:
                x = x.cuda().type(torch.cuda.FloatTensor)
                y = y.cuda().type(torch.cuda.FloatTensor)

            else:
                x = x.float()
                y = y.float()

            for i in range(15):
                # train on new game data
                self.network.optimize(x, y, batch_size=10000, iterations=1, alpha=0.0001)

                # test network
                prediction = self.network.forward(x)
                prediction = prediction[:,82]
                prediction[prediction>=0] = 1
                prediction[prediction<0] = -1
                actual = y[:,82]
                print(torch.sum(((prediction+actual)/2)**2)/y.shape[0], "%")

            print(100*torch.sum((actual+1)/2)/y.shape[0])
            # save model
            self.save_model(version=iter)

            # save game data
            self.save_data(x, y, version=iter)

            # clear memory
            del(x)
            del(y)
