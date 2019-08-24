import logging
import torch
from go_engine.go_engine import GoEngine
from teeny_go import TeenyGo


from __future__ import print_function
from dlgo import agent
from dlgo import goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move
import time


class GoTrainer(object):

    # go trainer takes go engine and teeny-go-ai
    # trains teeny go using self play
    # saves and loads teeny go models from model folder
    # saves games in a data folder seperated by model version
    # save log files with relevent information

    def __init__(self):
        pass

    def save_model(self):
        pass

    def save_game(self):
        pass

    def load_model(self):
        pass

    def load_game(self):
        pass

    def play_game(self):
        pass

    def train(self, num_games=100):

        # for number of games
        for i in range(num_games):
            pass
            # initialize game
            # play through game
            # save game data


        pass
