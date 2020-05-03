import torch
import pyspiel

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


class TeenyGo(object):

    def __init__(self, vn=None, pn=None):

        self.value_network = vn
        self.policy_network = pn

    def get_move(self, x):
        # get move tensor
        state_tensor = self.generate_state_tensor()
        state_tensor = torch.from_numpy(state_tensor).float()
        move_tensor = self.policy_network.forward(state_tensor)
        move_tensor = move_tensor.detach().numpy().reshape(-1)

        cl = self.value_net.forward(state_tensor)

        # remove invalid moves
        valid_moves = self.board_state.legal_actions_mask()
        valid_moves = np.array(valid_moves[0:441]).reshape(21, 21)
        valid_moves = valid_moves[1:10,1:10].reshape(81)
        valid_moves = np.append(valid_moves, 0)
        move_tensor = move_tensor * valid_moves

        moves = list(range(82))
        sum = np.sum(move_tensor[0:82])

        print("Confidence Level: {}".format(cl))

        if sum > 0:
            move = moves[np.argmax(move_tensor[0:82])]
            #move = 9*(move%9) + move//9
            #move = np.random.choice(moves, p=move_tensor[0:82]/sum)
            print("move:", move)
        else:
            print("ai: passed")
            move = 81

    def get_winrate(self, x):
        pass

    def mcts_step(self, x, width, depth):

        if depth == 0: return None

        # get policy
        p = self.policy_network.forward(x)

        # get best moves
        moves = self.get_best_moves(p, n)

        # get simulated moves
        sims = self.get_simulated_moves(moves)

        # test each sim
        for s in sims:
            self.mcts_step(s, width, depth-1)

    def get_best_moves(self, p, n):
        pass

    def get_simulated_moves(self, moves):
        pass
