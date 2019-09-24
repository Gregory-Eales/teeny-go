import numpy
#import pyspiel
import torch
import glob



class Reader(object):

    def __init__(self):

        # init translation dict
        self.letter_to_number = {}
        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

        # populate translation dict
        for i in range(9):
            self.letter_to_number[self.letters[i]] = i

    def reset_reader(self):

        # reset board

        pass


    def get_sgf_paths(self, path):
        pass

    def load_file(self, path):
        file = open(path, mode='r')

    def convert_file(self):
        pass

    def create_board_tensor(self):
        pass
