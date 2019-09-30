import logging
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Rater(object):

    def __init__(self, k_factor=32):

        # k factor is 16 for experts and 32 for weaker players
        # determines amount that scores are updated
        self.k_factor = 32
        self.file_name = "elo.csv"
        self.file_path = "data/"
        self.file = read_elo_file()

    def read_elo_file(self):
        return pd.load_csv(self.file_path+self.file_name)

    def initilize_file_path(self):
        # use os lib to find file path relative to execution path
        pass

    def update_score(self, A1, A2, winner):

        assert(type(A1) == string), "A1: agent type must string"
        assert(type(A2) == string), "A2: agent type must string"
        assert(type(winner) == string), "winner: winnner type must be string"

        # get current score for each agent in csv file
        A1_score = None
        A2_score = None

        # convert winner to numerical value
        winner = None

        # calculate updated score
        A1_update = self.calculate_updated_score(A1_score, A2_score, winner)
        A2_update = self.calculate_updated_score(A2_score, A1_score, winner)

        # write new elo rating to csv file, save

    def calculate_expected_score(self, R1, R2):
        return 1/(1+10**((R2-R1)/400))

    def calculate_updated_score(self, R1, R2, S):

        # R1 = score to be updated
        # R2 = opponent score
        # S = actual score
        # or percentage of wins against opponent

        expected_score = self.calculate_expected_score(R1, R2)
        return R1 + self.k_factor*(S - expected_score)

    def initilize_elo_file(self):
        # create new elo file

        # check to to see if elo file is already in place
        pass




def main():
    pass

if __name__ == "__main__":
    main()
