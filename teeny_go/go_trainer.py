import logging
import torch
from teeny_go_ai import TeenyGo
from go_engine.go_engine import GoEngine
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
        self.engine.new_game()
        playing = True
        move_counter = 0
        while playing:

            state = self.engine.get_board_tensor()
            state = torch.from_numpy(state).float()

            # get move from ai
            move = self.teeny_go.create_move_vector(state)


            deciding = True
            self.invalid_count = 0
            while deciding:
                counter += 1
                move = self.teeny_go.get_move()
                # check if move is valid
                #print(move)
                if move == "pass":
                    self.engine.change_turn()
                    deciding = False
                    self.pass_count += 1

                elif self.engine.check_valid(move) == True:
                    #print(self.teeny_go.last_move)
                    self.engine.make_move(move)
                    self.engine.change_turn()
                    deciding = False

                else:
                    self.invalid_count += 1

                if self.invalid_count > 81:
                    deciding = False
                    playing = False

            move_counter += 1
            if self.pass_count >= 2:
                playing = False

            if np.sum(np.where(self.engine.board != 0, 1, 0)) > 70:
                playing = False




        #print(self.teeny_go.move)

        self.engine.print_board()
        print(counter)
        print("Moves Played:",  move_counter)

    def play_optimized(self):

        # initilize new game
        self.engine.new_game()

        # main game loop
        while self.engine.is_playing:

            # get board tensor
            tensor = self.engine.get_board_tensor()
            # create move vector with tensor
            move = self.teeny_go.create_move_vector(tensor)
            # decide move loop
            while self.engine.is_deciding:

                # check if pass
                if self.engine.check_pass(move) == True: pass
                # check if move is valid
                if self.engine.check_valid(move) == True: pass

                # if move is valid store move in data cache


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
