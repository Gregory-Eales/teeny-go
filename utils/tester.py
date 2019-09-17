import logging

import numpy as np

from .agents.random_agent import RandomAgent
from .rater import Rater

class Tester(object):

    def __init__(self):

        self.random_agent = RandomAgent()
        self.rater = Rater()

        self.logger = logging.getLogger(name="Tester")

    def play_through_game(self, a1, a2, num_games):

        # play through n games agent 1 vs agent 2
        # update elo scores
        # save elo log
        self.logger.info("Agent1 win rate: {}%".format(win_rate))

    def play_through_random(self, agent):
        self.play_through_game(agent, self.random_agent)

    def calculate_sample_size(self, conf_lvl, choice_prob, conf_interv):
        return

    def test_outcome_prediction_accuracy(self, model, data):
        self.logger.info("{} prediction accuracy: {}%".formal(model, accuracy))

    def load_data(self, data_path):
        self.logger.info("data from {} loaded".format(data_path))

    def load_model(self, model):
        self.logger.info("Model {} loaded".format(model))
