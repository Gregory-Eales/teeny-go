import numpy as np
import gym
import argparse
import torch
import glob
from tqdm import tqdm
import time
import codecs



class Reader(object):

    def __init__(self):

        # init translation dict
        self.letter_to_number = {}
        self.letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "t"]

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

            for i, line in enumerate(lines):

                self.check_winner(line)

                self.add_sample(i, line)

            if save:

                try:
                    self.save_tensors(j, dest_path)

                except:
                    pass

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
                
                if line[0:3] == "AB[":
                    loc = line[3:5]

                else:   
                    loc = line[k+3:k+5]
                
                try:

                    if line[k:k+4] in [";B[]", ";W[]"]:
                        move = 81

                    else:
                        x = self.letters.index(loc[0])
                        y = self.letters.index(loc[1])

                        move = 9*(x) + (8-y)
                        if move > 80:
                            move = 81

                    state, reward, done, _ = self.env.step(move)
                    

                    move = self.generate_move(move)
                    self.move_tensor.append(move)
                    self.state_tensor.append([self.prev_state])
                    self.prev_state = state

                    break

                except:
                    break
    
    def save_tensors(self, j, dest_path):
        # convert tenors lists to tensors
        x = np.concatenate(self.state_tensor)
        y = np.concatenate(self.move_tensor)

        # convert numpy tensors to torch tensors
        x = torch.from_numpy(x).type(torch.int8)
        y = torch.from_numpy(y).type(torch.int8)

        # save tensors in data folder
        torch.save(x, "{}DataX{}{}".format(dest_path, j, ".pt"))
        torch.save(y, "{}DataY{}{}".format(dest_path, j, ".pt"))

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
