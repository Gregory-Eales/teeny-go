import numpy
#import pyspiel
import torch
import glob
from tqdm import tqdm



class Reader(object):

    def __init__(self):

        # init translation dict
        self.letter_to_number = {}
        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "t"]

        # populate translation dict
        for i in range(10):
            self.letter_to_number[self.letters[i]] = i

        # initilize game state
        self.initialize_game_state()

        self.move_map = self.get_move_map()

    def generate_date(self, paths, dest_path):

        # reset generator
        self.initialize_game_state()

        # loop through paths:
        for j, path in tqdm(enumurate(paths)):
        #   load file at path
            file = open(path, mode='r')
        #   read file
            lines = file.readlines()
        #   loop through lines
            for i, line in enumerate(lines):

                # get winner
                if i == 0:
                    outcome = line.split("RE[")[1].split("]")[0][0]
                    if outcome == "d":
                        winner = "draw"
                    elif outcome == "w":
                        winner = "white"
                    elif outcome == "b":
                        winner = "black"

                # get and translate move
                if i != 0:
                    loc = line[3:5]
                    try:
                        x = self.letter_to_number[loc[0]]
                        y = self.letter_to_number[loc[1]]
                        move = x + 9*y
                        if move == 90:
                            move = 81

                        self.update_board()
                        self.move_states.append(self.generate_move_tensor(move, winner))
                        self.games_states.append(self.generate_move_tensor())
                        # make move
                        move = self.move_map[move]


                    except: pass

            # convert tenors lists to tensors

            # convert numpy tensors to torch tensors

            # save tensors in data folder





        #   for move in moves in file:
        #       convert move from string to number
        #       make move
        #       make and save board tensor

        pass

    def update_board(self):
        state = self.board_state.observation_as_normalized_vector()
        state = np.array(state).reshape(-1, 81)
        state = (state[0] + state[1]*-1)
        self.game_states.append(np.copy(state.reshape(1, 9, 9)))

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
        self.move_states = []
        for i in range(7): self.game_states.append(np.zeros([9,9]))

    def generate_move_tensor(self, move, winner):
        turn = self.board_state.current_player()

        move_tensor = np.zeros([1, 82])

        if turn == 0 and winner == "black":
            outcome = 1

        elif turn == 1 and winner == "white":
            outcome = 1

        elif winner == "draw":
            outcome = 0

        else: outcome = -1

        move_tensor[0][81] = outcome

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

    def reset_reader(self):

        # reset board

        pass

    def save_game(self):
        pass

    def get_sgf_paths(self, path):
        pass

    def load_file(self, path):
        file = open(path, mode='r')

    def convert_file(self):
        pass

    def create_board_tensor(self):
        pass
