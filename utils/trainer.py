import logging
import time
import os

import torch
import pyspiel
import numpy as np

from multi_go_engine import MultiGoEngine



class GoTrainer(object):

    def __init__(self, network):

        # initialize model
        self.network = network

        # save network attributes
        self.num_res = self.network.num_res_blocks
        self.num_channels = self.netork.num_channels

        # set model name
        self.model_name = "Model-R{}-C{}".format(self.num_res, self.num_channels)

        # load game engine
        self.engine = MultiGoEngine()

    def save_model(self, version):
        path = "models/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filename = "Model-R{}-C{}-V{}.pt".format(self.num_res, self.num_channels, version)
        torch.save(self.network.state_dict(), path+filename))

    def load_model(self, version):
        path = "models/Model-R{}-C{}/".format(self.num_res, self.num_channels)
        filename = "Model-R{}-C{}-V{}.pt".format(self.num_res, self.num_channels, version)
        self.network.load_state_dict(torch.load(path+filename))

    def save_data(self):
        pass

    def load_data(self):
        pass

    def play_game(self):
        pass

    def get_game_data(self):
        pass

    def train(self):
        counter = 0
        t = time.time()
        hour = 60*60
        while (time.time()-t)/hour < 8:
            self.play_game()
            self.get_game_data()
            self.teeny_go.network.optimize(self.x_data[-1], self.y_data[-1], iterations=1, batch_size=self.x_data[-1].shape[0])
            torch.save(self.x_data[-1], os.path.join('data', "Model_R5_C64_DataX"+str(counter)+".pt"))
            torch.save(self.y_data[-1], os.path.join('data', "Model_R5_C64_DataY"+str(counter)+".pt"))
