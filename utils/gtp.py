import numpy as np
import gym
from gym_go.envs.go_env import GoEnv
import argparse
import sys



class GTP(object):


	def __init__(self):


		pass


	def tournament(self, bot1, bot2):

		pass



	def gtp_loop(self):

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
	

	