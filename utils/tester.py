import logging

import numpy as np

from .agents.random_agent import RandomAgent
from .rater import Rater

class Tester(object):

    def __init__(self):

        self.random_agent = RandomAgent()
        self.rater = Rater()

        # initialize logger
        self.logger = logging.getLogger(name="Model-Tester")

        # initilize game state
        self.initilize_game_state()

    def initialize_game_state(self):
        # create go board
        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = pyspiel.load_game("go", board_size)
        self.board_state = game.new_initial_state()
        self.game_states = []
        for i in range(7): self.game_states.append(np.zeros([9,9]))

    def generate_state_tensor(self):

        black = []
        white = []
        turn = self.board_state.current_player()

        if turn == 1:
            turn = [np.zeros([1, 9, 9])]

        elif turn == 0:
            turn = [np.ones([1, 9, 9])]

        for i in range(1, 6):
            black.append(np.copy(np.where(self.game_states[-i] == 1, 1, 0).reshape(1, 9, 9)))
            white.append(np.copy(np.where(self.game_states[-i] == -1, 1, 0).reshape(1, 9, 9)))

        black = np.concatenate(black, axis=0)
        white = np.concatenate(white, axis=0)
        turn = np.concatenate(turn, axis=0)

        output = np.concatenate([black, white, turn]).reshape(1, 11, 9, 9)

        return output

    def play_through_game(self, a1, a2, num_games=100):

        # play through n games agent 1 vs agent 2
        for n in range(num_games):

            # reset game board
            self.initilize_game_state()
            turn = "black"
            while not self.board_state.is_terminal():

                # AI 1 make
                if self.board_state.current_player() == 0:
                    self.make_ai_move(a1)

                # AI 2 make move
                else:
                    self.make_ai_move(a2)


        # update elo scores
        # save elo log
        self.logger.info("Agent1 win rate: {}%".format(win_rate))

    def make_ai_move(self, ai):
        pass

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
