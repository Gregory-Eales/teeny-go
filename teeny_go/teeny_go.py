import torch
import pyspiel

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


class TeenyGo(object):

    def __init__(self, vn=None, pn=None):

        self.value_network = vn
        self.policy_network = pn

    def get_move(self, x):
        pass

    def get_winrate(self, x):
        pass

    def mcts_step(self, x, width, depth):

        if depth == 0: return None

        # get policy
        p = self.policy_network.forward(x)

        # get best moves
        moves = self.get_best_moves(p, n)

        # get simulated moves
        sims = self.get_simulated_moves(self)

        # test each sim
        for s in sims:
            self.mcts_step(s, width, depth-1)

    def get_best_moves(self, p, n):
        pass

    def get_simulated_moves(self, moves):
        pass
