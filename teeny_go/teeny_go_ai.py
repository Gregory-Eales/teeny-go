import torch
from teeny_go_network import TeenyGoNetwork
import time

class TeenyGo(object):

    # Main Tasks:
    # - go engine
    # - go game gui

    # Teeny Go To Do:
    # - tree search method
    # - internal go engine for tree search
    # - internal board memory

    def __init__(self):


        self.board_state = torch.zeros(11, 9, 9) # should be (Nx2 + 1(turn state)) x 9 x 9
        self.network = TeenyGoNetwork()
        self.output_buffer = {"white":[], "black":[]}
        self.input_buffer = {"white":[], "black":[]}
        self.move = None
        self.last_move = None
        self.first_move = True
        self.first_vector = True
        self.turn = "black"

    def create_move_vector(self, x):

        if self.first_vector:
            self.first_vector = False

        else:
            self.output_buffer[self.turn][-1][0][self.last_move] = 2

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
            _, move = (self.output_buffer[self.turn][-1][0][0:81].max(dim=0))
            move = move.tolist()
            self.last_move = move
            move = [move%9, move//9]
            return move

        else:
            self.output_buffer[self.turn][-1][0][self.last_move] = 0
            _, move = (self.output_buffer[self.turn][-1][0][0:81].max(dim=0))
            move = move.tolist()
            self.last_move = move
            move = [move%9, move//9]
            return move

    def finalize_data_winner(self, winner):

        white_input = torch.cat(self.input_buffer["white"])
        white_output = torch.cat(self.input_buffer["white"])

        if winner == "black":

            # change white winner pred to -1
            self.input_buffer["white"]
            # change black winner pred to 1
            pass

        # change winner pred to -1
        else:
            pass


    def save_model(self):
        pass

    def load_weights(self, weight_path=None):
        pass
