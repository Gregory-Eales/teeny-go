import torch
from teeny_go_network import TeenyGoNetwork

class TeenyGo(object):

    # Main Tasks:
    # - go engine
    # - go game gui

    # Teeny Go To Do:
    # - tree search method
    # - get move method
    # - internal go engine for tree search
    # - internal board memory

    def __init__(self):


        self.board_state = torch.zeros(11, 9, 9) # should be (Nx2 + 1(turn state)) x 9 x 9

        self.network = TeenyGoNetwork()

    def make_move(self):

        # make value and policy prediction
        prediction = self.network.forward(self.board_state)
        policy, resign, value = prediction[0:81], prediction[-2], prediction[-1]

    def save_model(self):
        pass
