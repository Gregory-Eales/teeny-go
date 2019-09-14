import numpy as np
from agents.random_agent import RandomAgent
from rater import EloRater

class Tester(object):

    def __init__(self):

        self.random_agent = RandomAgent()
        

    def play_through_game(self, a1, a2, num_games):

        # play through n games agent 1 vs agent 2
        # update elo scores
        pass

    def play_through_random(self, agent):
        self.play_through_game(agent, self.random_agent)


    def calculate_sample_size(self, conf_lvl, choice_prob, conf_interv):
        return 
