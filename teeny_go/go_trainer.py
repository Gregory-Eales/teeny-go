import logging
import torch
from teeny_go import TeenyGo
from go_engine import GoEngine



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
        self.load_model(model_path="Models/"+self.model_name)

        # load game game engine
        self.engine = GoEngine()

    def save_model(self):
        torch.save(self.teeny_go.state_dict(), "Models/"+self.model_name)

    def load_model(self, model_path="Models/"+self.model_name):
        self.teeny_go.load_state_dict(torch.load("Models/"+self.model_name))

    def play_game(self):
        self.engine.new_game()
        playing = True
        while playing:

            # get move from ai
            move = self.teeny_go.network.forward()

            # check if move is valid
            if self.engine.check_valid(move) == True:
                self.engine.make_move(move)
                self.engine.change_turn()


    def train(self, num_games=100, iterations=10):

        for iter in range(iterations):
            for game in range(num_games):
                pass
                # play through game
                self.play_game()
                # save game data

            # shuffle game data, train

        pass

def main():
    game = goboard.GameState.new_game(9)
    print(game.board.)


if __name__ == "__main__":
    main()
