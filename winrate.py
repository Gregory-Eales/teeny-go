import subprocess
import time
from tqdm import tqdm
import argparse
import gym
import torch
import numpy as np
from random import shuffle, randint
import threading
import copy

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


def init_gnu_process(level=1):
    # runs command: gnugo --mode gtp --level 5
    player = subprocess.Popen(
        ["gnugo", "--mode", "gtp", "--level", str(level), "--komi", "7", "--depth", "1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True
    )
    # Send the initial 'boardsize' command to both processes
    send_command(player, "boardsize 9")
    return player

def init_go_env():
    parser = argparse.ArgumentParser(description='Go Simulation')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()
    return gym.make('gym_go:go-v0',size=args.boardsize, komi=5.5)#, reward_method='real')

    
def load_model():
    model = torch.load("/Users/greg/Repositories/teeny-go/pi-model-r12-c256-e1200.pt", map_location=torch.device('cpu'))
    print("loaded model")
    return model



def explore_node(model, state, env, curr_depth=0, breadth=3, max_depth=3):
    with torch.no_grad():
        if curr_depth == max_depth:
            _, value = model(state[0:4]).unsqueeze(0).float()
            return value.item()

        v = 0
        # get top k actions do explore
        action, value = model(state)
        action = action*torch.tensor(env.valid_moves()).float()
        action[0][81] = 0
        actions = torch.topk(action, breadth, dim=1).indices
        for a in actions:
            env_copy = copy.deepcopy(env)
            copy_state, _, _, _ = env_copy.step(a.item())
            v += explore_node(model, copy_state, env_copy, curr_depth+1, breadth=breadth, max_depth=max_depth)
        return v / breadth
    
    
def mcts(model, state, env, max_depth=3, breadth=3):
    '''monte carlo tree search'''
    with torch.no_grad():
        state = torch.tensor([state[0:4]]).float()
        action, _ = model(state)
        action = action*torch.tensor(env.valid_moves()).float()
        action[0][81] = 0
        actions = torch.topk(action, breadth, dim=1).indices

        best_move = None
        min_value = 2
        curr_player = env.turn()

        for a in actions:
            env_copy = copy.deepcopy(env)
            copy_state, _, _, _ = env_copy.step(a.item())
            # if distance is less than min_value then thats the best move
            value = explore_node(model, copy_state, env_copy, curr_depth=0, breadth=breadth, max_depth=max_depth)
            distance = abs(value - curr_player)
            if distance < min_value:
                min_value = distance
                best_move = a.item()
        return best_move
        


def get_action(model, state, env, max_depth=3, depth=0):

    # get current turn from env
    turn = env.turn()
    
    #  actions from policy
    action, value = model(state)

    # valid moves
    action = action*torch.tensor(env.valid_moves()).float()

    # set 81 to 0
    action[0][81] = 0

    # get the top 10 actions
    action = torch.topk(action, 3, dim=1).indices

    best_action = None
    best_dist = 2
    for i in range(len(action)):

        # make a copy of the env
        env_copy = copy.deepcopy(env)
        # take the action
        copy_state, _, _, _ = env_copy.step(action[0][i].item())
        # get the value of the new state
        _, new_value = model(torch.tensor(copy_state[0:4]).unsqueeze(0).float())
        new_value = new_value.item()


        distance = abs(new_value - turn)
        if distance < best_dist:
            best_dist = distance
            best_action = action[0][i].item()


    #print('curr likely winner is ', value.item())

    return best_action



def play_gnugo_game(model, level=1):
    
    # initialize a new gnu go process
    player = init_gnu_process(level=level)

    # init env
    env = init_go_env()

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "t"]
    # convert to dictionary that points from letter to index
    letter_to_index = {letters[i].upper(): i for i in range(len(letters))}
    # print(letter_to_index)

    iterator = iter(["B", "W"] * 200)
    move = None
    # reset the environment
    done, state  = False, env.reset()
    num_moves = 0
    while not done:
        num_moves += 1

        curr_player = next(iterator)

        if curr_player == "B":
            if move == 81:
                env.step(81)
                break

            with torch.no_grad():
                state = torch.tensor([state[0:4]]).float()
                #print(state.shape)
                action, _ = model(state)

                
                valid_move = torch.tensor(env.valid_moves()).float()
                action = action*valid_move


                # apply softmax to action
                action = torch.softmax(action, dim=1)

                #  # print the action as a list of probabilities
                # print('------------------------------------')
                # print([round(100*p, 2) for p in action[0].tolist()])
                # print('------------------------------------')
                
                # top action, 
                # randomly select top 3 actions, first get top 3 actions
                action = torch.topk(action, 2, dim=1).indices
                # # then randomly select one of the top 3
                move = action[0][np.random.randint(2)].item()

                # use function instead of random
                #move = get_action(model, state, env)
                #move = mcts(model, state, env, max_depth=3, breadth=3)

            if move == 81 or sum(env.valid_moves()) <= 1:
                str_move = 'pass'
                move = 81
            else:
                str_move = letters[move % 9] + str(9 - move // 9)
            send_command(player, "play " + curr_player + " " + str_move)

        else:

            send_command(player, "genmove " + curr_player)
            move = get_response(player)

            if move.lower() == "pass" or move[0].lower() == "r":
                move = 81
            else:
                move = letter_to_index[move[0]] + 9 * (9 - int(move[1]))

            # use env random move
            # move = 81
            # str_move = 'pass'
            # for i in range(10):
            #     move = env.uniform_random_action()
            #     if move != 81:
            #         str_move = letters[move % 9] + str(9 - move // 9)
            #         break

            #send_command(player, "play " + curr_player + " " + str_move)

        state, reward, done, info = env.step(move)

        if done:
            break

    #env.render(mode="terminal")
    # send command final_score
    send_command(player, "final_score")
    winner = get_response(player)

    # close the process
    player.stdin.close()

    # print(f"winner: {winner}")
    # print(f"num moves: {num_moves}")

    # if winner = 1 then print teeny-go won else
    if winner[0].lower() == 'b':
        env.render(mode="terminal")
        return 1
    return 0


def main():
    model = load_model()
    play_gnugo_game(model, level=1)
    num_wins = 0
    num_games = 10
    for i in tqdm(range(num_games)):
        try:
            num_wins += play_gnugo_game(model, level=1)
        except:
            pass
    print(f"num wins: {num_wins} out of {num_games} ({round(100*num_wins/num_games, 2)}%)")


    

if __name__ == "__main__":
    main()