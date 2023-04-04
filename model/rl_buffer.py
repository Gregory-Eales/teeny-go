import multiprocessing
import gym


class RLBuffer(object):


	def __init__(self):


		# number of games


	def generate_game(self):

        parser = argparse.ArgumentParser(description='Go Simulation')
        #parser.add_argument('--randai', action='store_true')
        parser.add_argument('--boardsize', type=int, default=9)
        args = parser.parse_args()

        go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='real')

        return go_env


	def step(self, actions):


		pass