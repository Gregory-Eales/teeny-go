import logging
import torch
from teeny_go import TeenyGo



#from go_engine.dlgo import agent
import time


class GoTrainer(object):

    # go trainer takes go engine and teeny-go-ai
    # trains teeny go using self play
    # saves and loads teeny go models from model folder
    # saves games in a data folder seperated by model version
    # save log files with relevent information

    def __init__(self, model_path=None):

        # initialize model
        self.teeny_go = TeenyGo()
        self.load_model(model_path=model_path)

    def save_model(self):
        pass

    def save_game(self):
        pass

    def load_model(self, model_path):
        self.teeny_go.load_weights(weight_path=model_path)

    def load_game(self):
        pass

    def play_game(self):
        board_size = 9
        game = goboard.GameState.new_game(board_size)

        while not game.is_over():



            bot_move = bots[game.next_player].select_move(game)

            game = game.apply_move(bot_move)

        # initilize board


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
