import torch
from teeny_go_network import TeenyGoNetwork

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
        self.output_buffer = []
        self.input_buffer = []
        self.move = None
        self.last_move = None
        self.first_move = True

    def create_move_vector(self, x):
        # make value and policy prediction
        self.first_move = True
        self.last_move = None
        prediction = self.network.forward(x)
        self.output_buffer.append(prediction)
        policy, pass_turn, value = prediction[0][0:81], prediction[0][-2], prediction[0][-1]
        _, p = policy.max(dim=0)

        if policy[p] > pass_turn:
            self.move = policy
        else:
            return "pass"

    def get_move(self):

        #print(self.last_move)
        if self.first_move == True:
            self.first_move = False
            _, move = (self.move.max(dim=0))
            move = move.tolist()
            self.last_move = move
            move = [move%9, move//9]
            return move

        else:
            self.move[self.last_move] = -1
            _, move = (self.move.max(dim=0))
            move = move.tolist()
            self.last_move = move
            move = [move%9, move//9]
            return move



    def save_model(self):
        pass

    def load_weights(self, weight_path=None):
        pass
