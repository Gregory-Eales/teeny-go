import logging

import torch
import numpy as np
import pyspiel
from tqdm import tqdm

from .agents.random_agent import RandomAgent
from .rater import Rater
from .multi_go_engine import MultiGoEngine


class Tester(object):

    def __init__(self):

        self.random_agent = RandomAgent()
        self.rater = Rater()

        # initialize logger
        self.logger = logging.getLogger(name="Model-Tester")

        # initilize game state
        self.initialize_game_state()

        self.move_map = self.get_move_map()

    def get_move_map(self):
        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = pyspiel.load_game("go", board_size)
        state = game.new_initial_state()
        return state.legal_actions()

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

    def update_board(self):
        state = self.board_state.observation_as_normalized_vector()
        state = np.array(state).reshape(-1, 81)
        state = (state[0] + state[1]*-1)
        self.game_states.append(np.copy(state.reshape(1, 9, 9)))

    def play_through_games(self, a1, a2, num_games=100):

        num_black_wins = 0
        num_white_wins = 0
        num_draws = 0

        # play through n games agent 1 vs agent 2
        for n in tqdm(range(num_games)):

            # reset game board
            self.initialize_game_state()
            turn = "black"
            while not self.board_state.is_terminal():

                # AI 1 make
                if self.board_state.current_player() == 0:
                    self.make_ai_move(a1)

                # AI 2 make move
                else:
                    self.make_ai_move(a2)

            score = self.board_state.returns()

            if score[0] == score[1]:
                num_draws += 1

            elif score[0] == 1:
                num_black_wins+=1

            elif score[1] == 1:
                num_white_wins+=1



        win_rate = num_black_wins/num_games
        print("Agent1 win rate: {}%".format(100*num_black_wins/num_games))
        print("Agent2 win rate: {}%".format(100*num_white_wins/num_games))
        print("Draw rate: {}%".format(100*num_draws/num_games))
        self.logger.info("Agent1 win rate: {}%".format(win_rate))

        return 100*num_black_wins/num_games


    def make_ai_move(self, ai):
        # get move tensor
        state_tensor = self.generate_state_tensor()
        state_tensor = torch.from_numpy(state_tensor).float()
        move_tensor = ai.forward(state_tensor)
        move_tensor = move_tensor.detach().numpy().reshape(-1)

        # remove invalid moves
        valid_moves = self.board_state.legal_actions_mask()
        valid_moves = np.array(valid_moves[0:441]).reshape(21, 21)
        valid_moves = valid_moves[1:10,1:10].reshape(81)
        valid_moves = np.append(valid_moves, 0)
        move_tensor[0:82] = (move_tensor[0:82] * valid_moves)
        moves = list(range(82))
        sum = np.sum(move_tensor[0:82])

        if sum > 0:
            move = np.random.choice(moves, p=move_tensor[0:82]/sum)
        else:
            move = 81


        self.logger.info("AI moved at: {}".format(move))

        self.board_state.apply_action(self.move_map[int(move)])

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

def main():
    mt = Tester()

if __name__ == "__main__":
    main()
