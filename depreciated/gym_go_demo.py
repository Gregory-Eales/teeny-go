import gym
import argparse
from matplotlib import pyplot as plt
import time


parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--randai', action='store_true')
parser.add_argument('--boardsize', type=int, default=9)
args = parser.parse_args()

go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')


rewards = []

def run_random(go_env):
	counter = 0
	done = False
	while not done:
		counter += 1
		#action = go_env.render('terminal')
		action = go_env.uniform_random_action()
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

	go_env.reset()
	return counter


parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--randai', action='store_true')
parser.add_argument('--boardsize', type=int, default=9)
args = parser.parse_args()

go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')

lengths = []
t = time.time()

for i in range(100):
	lengths.append(run_random(go_env))

print(time.time()-t)


print(lengths)

#plt.plot(rewards)
#plt.show()