import numpy as np
import gym
from gym_go.envs.go_env import GoEnv
import argparse
import sys

class RandomAgent(object):

	def __init__(self):

		self.name = "Random Agent"

		parser = argparse.ArgumentParser(description='Demo Go Environment')
		parser.add_argument('--randai', action='store_true')
		parser.add_argument('--boardsize', type=int, default=9)
		args = parser.parse_args()
		self.env = go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='real')

		self.env = GoEnv(size=args.boardsize)

	def make_move(self):
		valid_moves = self.env.get_valid_moves()
		move = np.random.choice(list(range(82)),p=valid_moves/np.sum(valid_moves))
		move = [move//9, move%9]
		return move
	
	def make_moves(self, valid_moves):
		moves = []
		for move in range(valid_moves):
			moves.append(self.make_move(move))
		return moves
		

def letter_to_coord(move):

	if move.lower() == 'pass':
		return str(81)

	letters = [char for char in "abcdefghj"]

	return [9-int(move[1]), letters.index(move[0].lower())]


def coord_to_letter(move):

	if move[0] == 9:
		return "PASS"

	letters = [char for char in "abcdefghj"]

	return letters[move[1]].upper() + str(9-move[0])



if __name__ == "__main__":

	agent = RandomAgent()

	"""
	parser = argparse.ArgumentParser(description='gtp')
	parser.add_argument('boardsize', , type=int, default='', help='')
	parser.add_argument('clear_board', , type=bool, default='', help='')
	parser.add_argument('komi', , type=float, default='', help='')
	parser.add_argument('play', , type=bool, default='', help='')
	parser.add_argument('genmove', , type=int, default='', help='')
	parser.add_argument('final_score', , type=int, default='', help='')
	parser.add_argument('quit', , type=int, default='', help='')
	parser.add_argument('name', , type=int, default='', help='')
	parser.add_argument('version', , type=int, default='', help='')
	parser.add_argument('known_command', , type=int, default='', help='')
	parser.add_argument('list_commands', , type=int, default='', help='')
	parser.add_argument('protocol_version', , type=int, default='', help='')
	
	"""

	all_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
					  'final_score', 'quit', 'name', 'version', 'known_command',
					  'list_commands', 'protocol_version', 'tsdebug']
	

	while True:

		command = input().strip().split()

		if command[0] == "quit":
			print('= \n\n')
			break

		elif command[0] == "protocol_version":
			print('=%s %s\n\n' % ('', 2,), end='')

		elif command[0] == "version":
			print('=%s %s\n\n' % ('', 1.0,), end='')

		elif command[0] == "name":
			print('=%s %s\n\n' % ('', agent.name,), end='')

		elif command[0] == "clear_board":
			agent.env.reset()
			print('= \n\n')
			

		elif command[0] == "list_commands":
			for i in all_commands:
				print('=%s %s\n\n' % ('', i,), end='')


		elif command[0] == "play":

			if command[2].lower() == 'pass':
				agent.env.step(81)

			else:
				agent.env.step(letter_to_coord(command[2]))

			print("= \n\n")

		elif command[0] == "genmove":
			
			try:
				move = agent.make_move()
				agent.env.step(move)

			except:
				move = [9, 0]
				agent.env.step(81)

			print('=%s %s\n\n' % ('', coord_to_letter(move),), end='')

		elif command[0] == "print_board":
			
			print(agent.env)

		else:
			print('= \n\n')


		command = []
			 
		sys.stdout.flush()