import subprocess
import time
from tqdm import tqdm
import argparse
import gym
import torch
import numpy as np
from random import shuffle, randint
import threading

# silence warnings
import warnings
warnings.filterwarnings("ignore")

# Function to send a command to a GNU Go process
def send_command(process, command):
    process.stdin.write(command + "\n")
    process.stdin.flush()

# Function to get a response from a GNU Go process
def get_response(process):
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("= "):
            return line[2:].strip()
        elif line.startswith("? "):
            return line[2:].strip()


def init_gnu_process(level=10):
    # runs command: gnugo --mode gtp --level 5
    player = subprocess.Popen(
        ["gnugo", "--mode", "gtp", "--level", str(level), "--komi", "5.5", "--play-out-aftermath"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True
    )
    # Send the initial 'boardsize' command to both processes
    send_command(player, "boardsize 9")
    return player

def init_go_env():
    parser = argparse.ArgumentParser(description='Go Simulation')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()
    return gym.make('gym_go:go-v0',size=args.boardsize, komi=5.5)#, reward_method='real')

    
def generate_sgf(handicap=0, level=10, filename="test.sgf"):

    player = subprocess.Popen(
        ["gnugo", "--mode", "gtp", "--level", str(level), "--outfile", filename]
        , stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True
    )


    send_command(player, "boardsize 9")
    
    iterator = iter(["B", "W"] * 100)
    prev_move = None

    for i in range(100):
        #time.sleep(1)  # Add a delay to be able to observe the game
        
        send_command(player, "genmove " + next(iterator))
        move = get_response(player)


        if move.lower() == "resign":
            print("p1 resigned. p2 wins!")
            break

        # if prev move and move are both pass, end game
        if prev_move == "PASS" and move == "PASS":
            print("game over")
            break

        prev_move = move

    send_command(player, "play B pass")
    send_command(player, "play W pass")
    send_command(player, "final_score")
    send_command(player, "quit")

    player.stdin.close()


def main():
    start_time = time.time()

    num_games = 100
    level = 10

    threads = []
    for i in tqdm(range(num_games // 20)):

        for j in range(20):
            t = threading.Thread(target=generate_sgf, args=(0, level, "games/test-" + str(i) + "-" + str(j) + ".sgf"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    print(f"it took {round(time.time() - start_time, 2)} seconds to generate {num_games} games @ lvl 10")  



def generate_gnu_tensors(level=10, game_num=0, path="data/"):
    
    # initialize a new gnu go process
    player = init_gnu_process(level=level)

    # init env
    env = init_go_env()

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "t"]
    # convert to dictionary that points from letter to index
    letter_to_index = {letters[i].upper(): i for i in range(len(letters))}
    # print(letter_to_index)

    iterator = iter(["B", "W"] * 200)
    # randomly generate a level between 1 and 10
    levels = [level, randint(1, 10)]
    shuffle(levels)
    level_iter = iter(levels * 200)


    # record moves
    states = []
    moves = []

    # reset the environment
    done, state  = False, env.reset()
    while not done:

        curr_level = next(level_iter)
        curr_player = next(iterator)

        send_command(player, "level " + str(curr_level))
        send_command(player, "genmove " + curr_player)

        move = get_response(player)

        if move.lower() == "pass" or move[0].lower() == "r":
            move = 81
        else:
            move = letter_to_index[move[0]] + 9 * (9 - int(move[1]))

            # a + 9 * (9 - b) = move
            # b = (move - a) / 9 + 9
        

        if True or curr_level == level:
            states.append(state[0] - state[1])
            move_array = np.zeros([1, 82])
            move_array[0][move] = 1
            moves.append(move_array)

        state, reward, done, info = env.step(move)

    # close the process
    player.stdin.close()

    # 1 = black, -1 = white, 0 = draw
    winner = env.winner()

    # make sure all are smallest signed integer type, int8
    states = torch.tensor(np.stack(states)).int()
    moves = torch.tensor(np.concatenate(moves, axis=0)).int()
    winners = torch.tensor([winner] * len(states)).unsqueeze(1).int()
    turns = torch.tensor([1 if i % 2 == 0 else -1 for i in range(len(states))]).unsqueeze(1).int()

    torch.save(states, "{}board-state-{}{}".format(path, game_num, ".pt"))
    torch.save(moves, "{}actions-{}{}".format(path, game_num, ".pt"))
    torch.save(winners, "{}winner-{}{}".format(path, game_num, ".pt"))
    torch.save(turns, "{}turns-{}{}".format(path, game_num, ".pt"))


def generate_gnu_dataset():

    
    num_games = 9000
    for i in tqdm(range(num_games//20)):
        threads = []

        for j in range(20):
            t = threading.Thread(target=generate_gnu_tensors, args=(10, 1000 + i*20 + j, "data/gnu-go-lvl-10-tensors/"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()
    

if __name__ == "__main__":
    #generate_gnu_tensors(level=10, game_num=0, path="data/gnu-go-lvl-10-tensors/")
    #generate_sgf(handicap=0, level=10, filename="test.sgf")
    generate_gnu_dataset()