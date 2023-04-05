import numpy as np
import gym
import argparse
import torch
import glob
from tqdm import tqdm
import time
import codecs

import warnings
warnings.filterwarnings("ignore")


class Reader(object):

    def __init__(self):

        # init translation dict
        self.letter_to_number = {}
        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "t"]
        self.populate_ranks()

        parser = argparse.ArgumentParser(description='Go Simulation')
        parser.add_argument('--boardsize', type=int, default=9)
        args = parser.parse_args()

        self.env = gym.make('gym_go:go-v0',size=args.boardsize, reward_method='real')

        self.winner = None
        self.completed = 0
        self.prev_state = None

    def reset(self):

        self.move_tensor = []
        self.state_tensor = []

        self.prev_state  = self.env.reset()

        self.winner = None

    def generate_data(self, paths, dest_path, save=True):

        for j in tqdm(range(len(paths))):

            file = open(paths[j], 'r+', encoding="utf-8")
            lines = [line.strip() for line in file.readlines()]
            self.reset()

            white_rank, black_rank = self.get_ranks(lines)

            # if we can't find the ranks then skip
            if not white_rank and not black_rank:
                continue

            # skip if both players are worse than 15k
            if white_rank<self.rank_dist.index('15k') and black_rank<self.rank_dist.index('15k'):
                continue

            for line in lines:
                if self.check_winner(line):
                    break

            for i, line in enumerate(lines):
                if self.add_sample(i, line):
                    break

            if save and len(self.move_tensor) > 0:
                self.save_tensors(j, dest_path)
            

    def populate_ranks(self):

        ranks_dist = []

        ranks_dist.append("?")

        for i in reversed(range(-10, 31)):

            ranks_dist.append("{}k".format(i))

        for i in range(1, 10):

            ranks_dist.append("{}d".format(i))

        self.rank_dist = ranks_dist

    def get_black_rank(self, lines):
        
        for line in lines:
            if line[0:2] == "BR":
                return line[3:-1]

    def get_white_rank(self, lines):
        
        for line in lines:

            if line[0:2] == "WR":
                return line[3:-1]

    def get_ranks(self, lines):
        
        try:

            white_rank = self.get_white_rank(lines)
            black_rank = self.get_black_rank(lines)

            white_rank = self.rank_dist.index(white_rank)
            black_rank = self.rank_dist.index(black_rank)

        except:
            black_rank = False
            white_rank = False

        return white_rank, black_rank

    def check_winner(self, line):

        # get winner
        if line[0:2] == "RE":
            
            outcome = line.split("RE[")[1].split("]")[0][0]
            
            if outcome == "D":
                self.winner = "draw"
                
            elif outcome == "W":
                self.winner = "white"
                
            elif outcome == "B":
                self.winner = "black"

    def add_sample(self, i, line):

        for k in range(len(line)):
            if line[k:k+3] in [";B[", ";W["] or line[k:k+3] == "AB[":

                turn = None
                if line[k:k+3] == ";B[":
                    turn = 'black'

                if line[k:k+3] == ";W[":
                    turn = 'white'
                
                if line[0:3] == "AB[":
                    loc = line[3:5]

                else:   
                    loc = line[k+3:k+5]
                
                if line[k:k+4] in [";B[]", ";W[]"]:
                    move = 81

                else:
                    x = self.letters.index(loc[0])
                    y = self.letters.index(loc[1])

                    move = x + y*9
                    if move > 80:
                        move = 81

                # some conflicts with sgf files and rules of go_env (making repeat moves)
                # going to quit out once this point is reached
                try:
                    state, reward, done, _ = self.env.step(move)
                except:
                    return True

                # if done then return true
                if done == 1:
                    return True


                # only save the data of the winning player
                if turn == self.winner or self.winner == 'draw':
                    move = self.generate_move(move)
                    self.move_tensor.append(move)
                    self.state_tensor.append([self.prev_state])
                    self.prev_state = state

               

    
    def save_tensors(self, j, dest_path):
        # convert tenors lists to tensors
        x = np.concatenate(self.state_tensor)
        y = np.concatenate(self.move_tensor)

        # convert numpy tensors to torch tensors
        x = torch.from_numpy(x).type(torch.int8)
        y = torch.from_numpy(y).type(torch.int8)

        # save tensors in data folder
        torch.save(x, "{}DataX{}{}".format(dest_path, self.completed, ".pt"))
        torch.save(y, "{}DataY{}{}".format(dest_path, self.completed, ".pt"))

        self.completed += 1
        

    def generate_move(self, move):

        move_array = np.zeros([1, 83])

        if self.winner == "black":
            move_array[0][-1] = 1

        if self.winner == "white":
            move_array[0][-1] = -1

        if self.winner == "draw":
            move_array[0][-1] = 0

        move_array[0][move] = 1

        return move_array

    def load_file(self, path):
        file = open(path, mode='r')

   

if __name__ == "__main__":

    """
    import gym
    import argparse

    parser = argparse.ArgumentParser(description='Go Simulation')
    #parser.add_argument('--randai', action='store_true')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()

    go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='real')

    print(go_env.turn())
    """
    t = time.time()
    reader = Reader()
    reader.generate_data(["test.sgf"], "")
    print(time.time()-t)
