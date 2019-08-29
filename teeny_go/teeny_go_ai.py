import torch
from teeny_go_network import TeenyGoNetwork
import time
import random

class TeenyGo(object):

    # Main Tasks:
    # - go engine
    # - go game gui

    # Teeny Go To Do:
    # - tree search method
    # - internal go engine for tree search
    # - internal board memory

    def __init__(self, num_channels=256, num_res_blocks=5):


        self.board_state = torch.zeros(11, 9, 9) # should be (Nx2 + 1(turn state)) x 9 x 9
        self.network = TeenyGoNetwork(num_channels=256, num_res_blocks=5)
        self.output_buffer = {"white":[], "black":[]}
        self.input_buffer = {"white":[], "black":[]}
        self.move = None
        self.last_move = None
        self.first_move = True
        self.first_vector = True
        self.turn = "black"
        self.random_cache = list(range(81))

    def create_move_vector(self, x):

        self.random_cache = list(range(81))

        if self.first_vector:
            self.first_vector = False

        else:
            pass
            #self.output_buffer[self.turn][-1][0][self.last_move] = 2


        # make value and policy prediction
        if x[0][-1][0][0] == 1:
            self.turn = "black"

        if x[0][-1][0][0] == 0:
            self.turn = "white"

        self.first_move = True
        self.last_move = None

        prediction = self.network.forward(x)

        self.output_buffer[self.turn].append(prediction)
        self.input_buffer[self.turn].append(x)


    def get_move(self):

        _ , p = self.output_buffer[self.turn][-1][0][0:81].max(dim=0)

        prediction = self.output_buffer[self.turn][-1]

        if prediction[0][0:81][p] < prediction[0][-2]:
            return "pass"

        if self.first_move == True:
            self.first_move = False

            if random.randint(0, 100) > 99 and self.random_cache != 0:
                pass
                move = random.choice(self.random_cache)

            else:
                _, move = (self.output_buffer[self.turn][-1][0][0:81].max(dim=0))
                move = move.tolist()
            self.last_move = move
            move = [move%9, move//9]
            return move

        else:

            try:
                pass
                self.random_cache.remove(self.last_move)

            except: pass

            if random.randint(0, 100) > 99 and self.random_cache != 0:
                pass
                move = random.choice(self.random_cache)
            else:
                _, move = (self.output_buffer[self.turn][-1][0][0:81].max(dim=0))
                move = move.tolist()
            self.last_move = move
            move = [move%9, move//9]
            return move

    def finalize_move(self):
        self.output_buffer[self.turn][-1][0][self.last_move] = 2

    def finalize_data_winner(self, winner):

        self.first_vector = True

        white_input = torch.cat(self.input_buffer["white"])
        white_output = torch.cat(self.output_buffer["white"])

        black_input = torch.cat(self.input_buffer["black"])
        black_output = torch.cat(self.output_buffer["black"])

        if winner == "black":

            # change white winner pred to -1
            white_output[:, 82] = -1
            white_output[white_output==2] = 0.5
            # change black winner pred to 1
            black_output[:, 82] = 1
            black_output[black_output==2] = 1

        else:
            white_output[:, 82] = 1
            white_output[white_output==2] = 1
            # change black winner pred to 1
            black_output[:, 82] = -1
            black_output[black_output==2] = 0.5


        x_data = torch.cat([white_input, black_input])
        y_data = torch.cat([white_output, black_output])

        # reset buffer
        self.output_buffer["white"] = []
        self.output_buffer["black"] = []
        self.input_buffer["white"] = []
        self.input_buffer["black"] = []

        return x_data, y_data


    def save_model(self):
        pass

    def load_weights(self, weight_path=None):
        pass
