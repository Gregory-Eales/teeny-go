import argparse
import gym
import threading
from multiprocessing import Process
import time
from tqdm import tqdm
from copy import copy


# Arguments
parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--randai', action='store_true', default=True)
parser.add_argument('--boardsize', type=int, default=9)
parser.add_argument('--komi', type=float, default=0)

args = parser.parse_args()

# Initialize environment
go_env = gym.make('gym_go:go-v0', size=args.boardsize, komi=args.komi)


def time_it(f, env, num_games):

	t = time.time()

	f(env, num_games)

	print("Took", time.time() - t, "seconds")

def simulate_games(env, num_games):


	for i in range(num_games):
		done = False
		while not done:
			action = go_env.uniform_random_action()
			_, _, done, _ = go_env.step(action)
		go_env.reset()

	print("Done!")

if __name__ == '__main__':

	num_games = 100

	time_it(simulate_games, copy(go_env), num_games=num_games)

	processes = []

	t = time.time()
	for i in range(5):

		processes.append(Process(target=simulate_games, args=(copy(go_env), num_games)))
		processes[-1].start()
	
	for i in range(5): processes[i].join()

	print("Took", time.time() - t, "seconds")