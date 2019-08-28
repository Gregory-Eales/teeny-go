import logging
import torch
from teeny_go_ai import TeenyGo

from go_engine.cython_go_engine import GoEngine

#from go_engine.go_engine import GoEngine
import numpy as np


#from go_engine.dlgo import agent
import time


class GoTrainer(object):

    # go trainer takes go engine and teeny-go-ai
    # trains teeny go using self play
    # saves and loads teeny go models from model folder
    # saves games in a data folder seperated by model version
    # save log files with relevent information

    def __init__(self, model_name="Model-V0"):

        self.model_name = model_name

        # initialize model
        self.teeny_go = TeenyGo()
        #self.load_model(model_path="Models/"+self.model_name)

        # load game game engine
        self.engine = GoEngine()

        self.pass_count = 0
        self.invalid_count = 0

        self.x_data = []
        self.y_data = []



    def save_model(self):
        torch.save(self.teeny_go.network.state_dict(), "Models/"+self.model_name)

    def load_model(self, model_path="Models/Model-V0"):
        self.teeny_go.network.load_state_dict(torch.load("Models/"+self.model_name))

    def play_game(self):

        counter = 0

        # reset game states
        self.engine.new_game()

        # main game loop
        while self.engine.is_playing:

            counter+= 1
            # get game vector
            move = self.get_move_vector()

            # reset turn
            self.reset_turn()

            # decide next move
            while self.engine.is_deciding:

                # take next game step
                self.take_game_step()

                # check if invalid limit exceeded
                self.check_invalid()

        # print final board
        self.engine.print_board()
        print(self.engine.score_game())
        print("black:", self.engine.black_score)
        print("white:", self.engine.white_score)
        print("Number of loops: ", counter)

    def reset_turn(self):
        self.invalid_count = 0
        self.engine.is_deciding = True

    def take_game_step(self):
        move = self.teeny_go.get_move()
        if self.engine.check_valid(move) == False:
            self.invalid_count += 1

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
        pass

    def train_all(self, num_games=100, iterations=10):

        for iter in range(iterations):
            for game in range(num_games):
                pass
                # play through game
                self.play_game()
                # save game data

            # shuffle game data, train
        pass

def main():
    gt = GoTrainer()
    gt.play_game()



if __name__ == "__main__":
    main()
