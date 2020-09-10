import numpy as np
from copy import copy
import argparse
import gym
import time


class MiniMax(object):

	def __init__(self, depth = 2):
		self.depth = depth

	def make_move(self, board):
		
		move = self.tree_search(board, self.depth)

	def tree_search(self, board, depth):

		if depth > 0:

			valid_moves = board.get_valid_moves()
			moves = np.argwhere(valid_moves).T[0].tolist()

			# Black = 0, White = 1
			turn = board.turn()
			sims = []
			for m in moves:
				sims.append(copy(board))

			for sim in sims:
				self.tree_search(sim, depth-1)
		
		else:
			return board.get_reward()

def main():


	minimax = MiniMax()

	parser = argparse.ArgumentParser(description='Demo Go Environment')
	parser.add_argument('--randai', action='store_true')
	parser.add_argument('--boardsize', type=int, default=9)
	args = parser.parse_args()

	go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')
	done = False
	rewards = []

	#while not done:
	for i in range(40):


		t = time.time()
		action = minimax.make_move(go_env)
		print("Time Taken:", time.time() - t)

		go_env.render("human")

		try:
			state, reward, done, _ = go_env.step(action)
		except Exception as e:
			print(e)
			continue
		if True:
			if go_env.game_ended():
				break
			action = go_env.uniform_random_action()
			state, reward, done, _ = go_env.step(action)
		

	else: print("Teeny-Go Lost!")

	
	if go_env.get_winning() == 1:
			print("Minimax Won!")
			
	else:
		print("MiniMax Lost!")

if __name__ == "__main__":
	main()