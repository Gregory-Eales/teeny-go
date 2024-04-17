import os
import pandas as pd
import numpy as np
import gym
import argparse
import torch
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
import re   
import json
import hashlib

WHITE = -1
BLACK = 1
EMPTY = 0
DRAW = 0

def make_env():
    parser = argparse.ArgumentParser(description='Go Simulation')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()
    return gym.make('gym_go:go-v0',size=args.boardsize, reward_method='real')


def get_content(str):
    return re.search(r'\[(.*?)\]', str).group(1)


def generate_dataset(source_path, dest_path, num_games):

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "t"]
    letter_to_number = {letters[i]: i for i in range(len(letters))}
    file_names = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source_path) for f in filenames if f.endswith('.sgf')][0:num_games]
    metadatas = []

    num_completed = 0
    num_duplicates = 0

    # create a set of hashes to check for duplicates
    hashes = set()

    for n in tqdm(range(num_games)):

        env = make_env()
        prev_state = env.reset()

        metadata = {'num_moves': 0, 'num_passes': 0, 'black_moves': 0, 'white_moves':0}

        actions = []
        rewards = []
        states = []
        winner = None
        done = False
        
        movestr = ""

        file = open(file_names[n], 'r+', encoding="utf-8").read()

        # loop through like a string
        for idx in range(len(file)):

            # dont't process games with manual placement
            if file[idx:idx+3] == "AB[":
                break

            if file[idx:idx+3] == "RE[":
                metadata["result"] = get_content(file[idx:])
                if file[idx+4] == "W":
                    winner = WHITE
                elif file[idx+4] == "B":
                    winner = BLACK
                else:
                    winner = DRAW

                # if last two digits are +T then timeout, break
                if metadata["result"][-2:] == "+T":
                    break

            if file[idx:idx+3] == "PB[" or file[idx:idx+3] == "PW[":
                name_color = "black_name" if file[idx:idx+3] == "PB[" else "white_name"
                metadata[name_color] = get_content(file[idx:])

            # BR or WR
            if file[idx:idx+3] == "BR[" or file[idx:idx+3] == "WR[":
                name_color = "black_rank" if file[idx:idx+3] == "BR[" else "white_rank"
                metadata[name_color] = get_content(file[idx:])

            ## ff - sfg_version
            if file[idx:idx+3] == "FF[":
                metadata["sgf_version"] = get_content(file[idx:])

            # same km
            if file[idx:idx+3] == "KM[":
                metadata["komi"] = get_content(file[idx:])

            # ru
            if file[idx:idx+3] == "RU[":
                metadata["rules"] = get_content(file[idx:])
            
            # tm time limit
            if file[idx:idx+3] == "TM[":
                metadata["time_limit"] = get_content(file[idx:])

            
            # move
            if file[idx:idx+3] == ";B[" or file[idx:idx+3] == ";W[":
                metadata["num_moves"] += 1
                move = file[idx+3:idx+5]
                movestr += file[idx+1:idx+5]
                #print([file[idx:idx+3], move, file[idx:idx+10]])
                turn = BLACK if file[idx:idx+3] == ";B[" else WHITE

                if turn == BLACK:
                    metadata["black_moves"] += 1
                else:
                    metadata["white_moves"] += 1

                if move[0] == "]" or move == "":
                    move = 81
                else:
                    x = letter_to_number[move[0]]
                    y = letter_to_number[move[1]]
                    move = x + y*9

                try:
                    state, _, done, _ = env.step(move)
                except:
                    env.render(mode="terminal")
                    print(f"Error in game {n} on move {metadata['num_moves']}")
                    print(f'attempted move: {move} - {[file[idx:idx+10]]}')
                    print('file name:', file_names[n])
                    return
                # don't record pass moves
                if move == 81:
                    metadata["num_passes"] += 1
                    prev_state = state
                    continue

                reward = 0
                if turn == winner:
                    reward = 1
                elif winner * -1 == turn:
                    reward = -1

                actions.append(generate_move(move))
                rewards.append([reward])
                states.append([prev_state[0:4]])
                prev_state = state

            if done:
                break

        # don't want to record entire move string in set
        metadata["moves_md5_hash"] = hashlib.md5(movestr.encode()).hexdigest()

        metadatas.append(metadata)

        if metadata["moves_md5_hash"] in hashes:
            num_duplicates += 1
            continue

        if metadata["num_moves"] < 20:
            continue
        
        save_tensors(num_completed, states, actions, rewards, dest_path)
        hashes.add(metadata["moves_md5_hash"])
        num_completed += 1

    pd.DataFrame(metadatas).to_csv("data/metadata.csv", index=False)
    print('\n' + '-'*50)
    print(f"Completed {num_completed} games out of {num_games} ({round(100*num_completed/num_games, 2)}%)")
    print(f"Skipped {num_duplicates} duplicate games {round(100*num_duplicates/num_games, 2)}%")
    print('-'*50 + '\n')
    

def save_tensors(sample_num, states, actions, rewards, path):
    # convert tenors lists to tensors
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    rewards = np.concatenate(rewards)

    # convert numpy tensors to torch tensors
    states = torch.from_numpy(states).type(torch.int8)
    actions = torch.from_numpy(actions).type(torch.int8)
    rewards = torch.from_numpy(rewards).type(torch.int8)

    # save tensors in data folder
    torch.save(states, f"{path}state-{sample_num}.pt")
    torch.save(actions, f"{path}action-{sample_num}.pt")
    torch.save(rewards, f"{path}reward-{sample_num}.pt")
        

def generate_move(move):
    move_array = np.zeros([1, 82])
    move_array[0][move] = 1
    return move_array
        
        
def main():
    path = 'data/ogs-high-quality/'
    generate_dataset(path, "data/ogs-quality-tensors/", 10000)

if __name__ == "__main__":
    main()