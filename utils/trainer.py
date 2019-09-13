import logging
import time
import os

import torch
import pyspiel
import numpy as np



class GoTrainer(object):

    def __init__(self, network):

        # initialize model
        self.network = network

        # save network attributes
        self.num_res = self.network.num_res_blocks
        self.num_channels = self.netork.num_channels

        # set model name
        self.model_name = "Model-R{}-C{}-V{}".format(1,1)

        # load game engine
        self.engine = MultiGoEngine()

    def save_model(self):
        torch.save(self.teeny_go.network.state_dict(),  os.path.join('models', model_path+".pt"))

    def load_model(self, model_path="Models/Model-V0"):
        self.teeny_go.network.load_state_dict(torch.load("Models/"+self.model_name))

    def play_game(self):
        pass

    def get_game_data(self):
        pass

    def check_invalid(self):
        if self.invalid_count > 81:
            print("Number of Invalid Moves:", self.invalid_count)
            self.engine.is_deciding = False
            self.engine.is_playing = False

    def get_move_vector(self):
        state = self.get_board_tensor()
        return self.teeny_go.create_move_vector(state)

    def get_board_tensor(self):
        state = self.engine.get_board_tensor()
        return torch.from_numpy(state).float()


    def creat_y_tensor(self, n):
        t = torch.zeros(83)

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
