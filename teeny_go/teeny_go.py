import torch
import pyspiel
import gym
import argparse
from matplotlib import pyplot as plt

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


class TeenyGo(object):

    def __init__(self, vn=None, pn=None, mcts_width=5, mcts_depth=1):

        self.value_network = vn
        self.policy_network = pn

        self.mcts_width = mcts_width
        self.mcts_depth = mcts_depth

    def generate_game(self):

        parser = argparse.ArgumentParser(description='Go Simulation')
        #parser.add_argument('--randai', action='store_true')
        parser.add_argument('--boardsize', type=int, default=9)
        args = parser.parse_args()

        go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')

        return go_env

    def get_move(self, state):
        
        move_tensor = self.policy_network.forward(state[0:3])
        move_tensor = move_tensor.detach().numpy().reshape(-1)

        cl = self.value_net.forward(state_tensor)

        # remove invalid moves
        valid_moves = state[3].reshape(1, 82)
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

    def get_state(self, board_sim)

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
        sims = self.get_simulated_moves(moves)

        # test each sim
        for s in sims:
            self.mcts_step(s, width, depth-1)

    def get_best_moves(self, p, n):
        pass

    def get_simulated_moves(self, moves):
        pass

def main():

    teeny_go = TeenyGo()

    parser = argparse.ArgumentParser(description='Demo Go Environment')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()
    go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')
    done = False
    rewards = []

    state = go_env.reset()

    while not done:

        #action = go_env.render('terminal')
        action = teeny_go.act(state)
        try:
            state, reward, done, _ = go_env.step(action)
        except Exception as e:
            print(e)
            continue
        if args.randai:
            if go_env.game_ended():
                break
            action = go_env.uniform_random_action()
            state, reward, done, _ = go_env.step(action)

        rewards.append(reward)

if __name__ == "__main__":
    main()
