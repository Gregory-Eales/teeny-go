from matplotlib import pyplot as plt
import numpy as np

import logging


class Rater(object):

    def __init__(self):

        # k factor is 16 for experts and 32 for weaker players
        self.k_factor = 32

    def calculate_expected_score(self, R1, R2):
        return 1/(1+10**((R2-R1)/400))

    def update_score(self, R1, R2, S):

        # R1 = score to be updated
        # R2 = opponent score
        # S = actual score
        # or percentage of wins against opponent

        expected_score = self.calculate_expected_score(R1, R2)
        return R1 + self.k_factor*(S - expected_score)




def main():
    pass

if __name__ == "__main__":
    main()
