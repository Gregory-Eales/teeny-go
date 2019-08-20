import torch
from evaluation_network import EvaluationNetwork
from value_network import ValueNetwork

class TeenyGo(object):


    # - network to pick best move
    # - network to predict winner
    # - tree search method
    # - get move method
    # - internal go engine for tree search
    # - internal board memory




    def __init__(self):

        self.value_network = ValueNetwork()
        self.advantage_network = EvaluationNetwork()

    def make_move(self):
        pass

    def save(self):
        pass
