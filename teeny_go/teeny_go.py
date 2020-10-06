import torch
#import pyspiel
import gym
import argparse
from matplotlib import pyplot as plt
from copy import copy
from copy import deepcopy
import numpy as np
import time
import pytorch_lightning as pl

#from .policy_network import PolicyNetwork
#from .value_network import ValueNetwork
from joint_network import JointNetwork 


class TeenyGo(pl.LightningModule):

    def __init__(self, vn=None, pn=None, jn=None, mcts_width=5, mcts_depth=1):

        self.value_network = vn
        self.policy_network = pn
        self.joint_network = jn

        self.mcts_width = mcts_width
        self.mcts_depth = mcts_depth

    def generate_game(self):

        parser = argparse.ArgumentParser(description='Go Simulation')
        #parser.add_argument('--randai', action='store_true')
        parser.add_argument('--boardsize', type=int, default=9)
        args = parser.parse_args()

        go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='real')

        return go_env

    def copy_game(self, game):
        return copy(game)

    def get_move(self, state, valid_moves):
        p, v = self.joint_network.forward(state)
        p = p.detach().numpy()
        print("Best Uncorrected Move: ", np.argmax(p))
        v = v.detach().numpy()
        p = p*valid_moves
        return np.argmax(p), v


    def get_state(self, board_sim):

        pass

    def get_winrate(self, x):
        pass

    def mcts(self, game, width, depth, first=True):


        x = game.get_state()[0:3].reshape([1, 3, 9, 9])

        if depth == 0:
            p, v = self.joint_network.forward(x).detach().numpy()
            return v

        # get policy
        p, v = joint_network.forward(x).detach().numpy()

        # remove invalid moves
        p = p*game.get_valid_moves()

        # get best moves
        moves = self.get_best_moves(p, width)

        # get simulated moves
        sims = self.get_simulated_games(game, moves)

        values = []
        # test each sim
        for sim in sims:
            values.append(self.mcts(sim, width, depth-1, first=False))

        if first == False:
            return np.sum(values)/len(values)

        if first == True:
            return moves[np.argmax(values)]

    def get_best_moves(self, p, width):
        
        moves = []

        for i in range(width):
            moves.append(np.argmax(p))
            p[0][moves[-1]] = 0

        return moves

    def get_best_single_move(self, board, n_moves=10):

        state = board.get_state()[0:3].reshape([1, 3, 9, 9])

        valid_moves = board.get_valid_moves()
        p, v_init = self.joint_network.forward(state)
        p = p.detach().numpy()
        p = p*valid_moves
        moves = self.get_best_moves(p, n_moves)

        print("Got moves")
        sims = []
        values = []
        for i, move in enumerate(moves):
            sims.append(copy(board))
            state, reward, done, _ = sims[-1].step(move)
            state = state[0:3].reshape([1, 3, 9, 9])
            """
            if state[0][2][0][0] == 1:
                state[0][2]*= 0
            else:
                state[0][2]+= 1
            """
            p, v = self.joint_network.forward(state)
            values.append(v.detach().numpy()[0][0])
        print("Value @ Move:", values)
        print("Simulated Games")
        #print(values)
        return moves[np.argmax(values)], v_init.detach().numpy()[0][0]

            

    def get_simulated_games(self, game, moves):
        
        sims = []

        for move in moves:
            sims.append(self.copy_game(game))
            sims[-1].step(move)

        return sims

def main():

    #policy_net = PolicyNetwork(alpha = 0.001,num_res=3, num_channel=128)
    #policy_net.load_state_dict(torch.load("PN-R3-C128-PFinal.pt", map_location={'cuda:0': 'cpu'}))
    #value_net = ValueNetwork(alpha = 0.001, num_res=2, num_channel=64)
    #value_net.load_state_dict(torch.load("VN-R2-C64-V4.pt", map_location={'cuda:0': 'cpu'}))

    class FakeObject(object):
        
        def __init__(self):
            self.params = {"lr":0.01, "layers":3}#, "activation":"sigmoid"}
        
        def __iter__(self):
            yield self.params.keys()
    
    fake_args = FakeObject()
    
    fake_args.in_channels = 3
    fake_args.kernal_size = 3
    fake_args.num_channels = 512
    fake_args.num_res_blocks = 1
    
    fake_args.gpu = 0 
    fake_args.early_stopping=True
    fake_args.max_epochs=200
    fake_args.batch_size=64
    fake_args.lr=1e-2
    fake_args.accumulate_grad_batches=64
    fake_args.check_val_every_n_epoch = 1
   
    # dataset params
    fake_args.num_games=1000
    fake_args.data_split=[0.9, 0.05, 0.05]
    fake_args.data_path="/kaggle/input/godataset/new_ogs_tensor_games/"

    joint_net = JointNetwork(fake_args)
    joint_net.load_state_dict(torch.load("./models/joint_model_v10.pt"))



    teeny_go = TeenyGo(pn=None, vn=None, jn=joint_net)

    parser = argparse.ArgumentParser(description='Demo Go Environment')
    parser.add_argument('--randai', action='store_true')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()

    wins = 0

    value_guesses = []
    for i in range(1):

        go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')
        done = False

        rewards = []

        state = go_env.reset()


        #while not done:
        for i in range(40):


            """
            state = go_env.get_state()[0:3].reshape([1, 3, 9, 9])
            valid_moves = go_env.get_valid_moves()
            t = time.time()
            #action = go_env.render('terminal')
            action = teeny_go.get_move(state, valid_moves)

              
            print(value_net.forward(state))
            print("Time:", time.time()-t)
            """
            
            """
            state = go_env.get_state()[0:3].reshape([1, 3, 9, 9])
            valid_moves = go_env.get_valid_moves()
            action = teeny_go.get_move(state, valid_moves)
            """
            
            
            
            

            #action = teeny_go.mcts(go_env, 3, 3)


            
                

            
            
            state = go_env.get_state()[0:3].reshape([1, 3, 9, 9])
            
            
            """
            if state[0][2][0][0] == 1:
                state[0][2]*= 0
            else:
                state[0][2]+= 1
            
            """
            

            
            valid_moves = go_env.get_valid_moves()
            action, v= teeny_go.get_move(state, valid_moves)
            v = v[0][0]
            
            
            
            
            #action, v = teeny_go.get_best_single_move(go_env, n_moves=5)
            

            value_guesses.append(v)

            print("tg action:", action)
            print("value prediciton:", v)

            try:
                state, reward, done, _ = go_env.step(action)
                """
                s = state[0:3].reshape([1, 3, 9, 9])

                if state[2][0][0] == 1:
                    state[2]*= 0
                else:
                    state[2]+= 1
                #print(s)
                """
                print(state[0:3])
                

            except Exception as e:
                print(e)
                continue
            

            if True:

                if go_env.game_ended():
                    break
                
                action = go_env.render("human")
                #action = go_env.uniform_random_action()
                state, reward, done, _ = go_env.step(action)
                print(state[0:3])


        if go_env.get_winning() == 1:
            print("Teeny-Go Won!")
            wins += 1

        else: print("Teeny-Go Lost!")

    print("Teeny-Go Won {} Games".format(wins))

    print(value_guesses)
    plt.plot(value_guesses)
    plt.show()
        

if __name__ == "__main__":
    main()
