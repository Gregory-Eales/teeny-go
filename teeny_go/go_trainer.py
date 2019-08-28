import logging
import torch
from teeny_go_ai import TeenyGo
from go_engine.cython_go_engine import GoEngine
import numpy as np
import time
import os


class GoTrainer(object):

    # go trainer takes go engine and teeny-go-ai
    # trains teeny go using self play
    # saves and loads teeny go models from model folder
    # saves games in a data folder seperated by model version
    # save log files with relevent information

    def __init__(self, model_name="Model-V0"):

        self.model_name = model_name

        # initialize model
        self.teeny_go = TeenyGo(num_channels=64, num_res_blocks=5)
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

            self.teeny_go.finalize_move()

        self.engine.print_board()

    def get_game_data(self):
        winner = self.engine.score_game()
        print(winner)
        x, y = self.teeny_go.finalize_data_winner(winner)
        self.x_data.append(x)
        self.y_data.append(y)

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
        counter = 0
        t = time.time()
        while (time.time()-t) < 60:
            counter+=1
            self.play_game()
            self.get_game_data()
            self.teeny_go.network.optimize(self.x_data[-1], self.y_data[-1], iterations=1, batch_size=self.x_data[-1].shape[0])
            torch.save(self.x_data[-1], os.path.join('data', "Model_R5_C64_DataX"+str(counter)+".pt"))
            torch.save(self.y_data[-1], os.path.join('data', "Model_R5_C64_DataY"+str(counter)+".pt"))
            self.x_data = []
            self.y_data = []
        #x_data = torch.cat(self.x_data)
        #y_data = torch.cat(self.y_data)


        #self.teeny_go.network.optimize(x_data, y_data, iterations=5)

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
    gt.train()



if __name__ == "__main__":
    main()
