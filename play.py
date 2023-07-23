import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from argparse import ArgumentParser, Namespace
import argparse
import gym


from model.teeny_go import TeenyGo
from model.policy_network import PolicyNetwork
from model.value_network import ValueNetwork
from model.joint_network import JointNetwork 



def play(args):

	net = JointNetwork(args)
	net = net.load_from_checkpoint(args.checkpoint_path, hparams=args)
	teeny_go = TeenyGo(pn=None, vn=None, jn=net)

	parser = argparse.ArgumentParser(description='Go Env')
	parser.add_argument('--randai', action='store_true')
	parser.add_argument('--boardsize', type=int, default=9)
	args = parser.parse_args()
	go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')
	done = False

	state = go_env.reset()

	#while not done:
	for i in range(80):

		
		action, value = teeny_go.get_move(state, go_env.valid_moves())

		action = int(action)

		state, reward, done, _ = go_env.step(action)
		go_env.done = False

		if go_env.game_ended():
			break
		go_env.render(mode='terminal')
		while True:
			try:
			
				action = input("enter your move: ").split(',')
				action = int(action[0])*9 + int(action[1])
				state, reward, done, _ = go_env.step(action)
				break
			
			except:
				print("invalid move")
	

if __name__ == '__main__':
	
	torch.manual_seed(0)
	np.random.seed(0)
	parser = ArgumentParser()

	# training params
	parser.add_argument("--gpu", type=int, default=0, help="number of gpus")
	parser.add_argument("--early_stopping", type=bool, default=True, help="whether early stopping is used or not")
	parser.add_argument("--max_epochs", type=int, default=100, help="number of epochs")
	parser.add_argument("--batch_size", type=int, default=16, help="size of training batch")
	parser.add_argument("--lr", type=int, default=1e-5, help="learning rate")
	#parser.add_argument("--accumulate_grad_batches", type=int, default=64, help="grad batches")
	parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="")
	parser.add_argument("--auto_lr_find", type=bool, default=False, help="finds the optimal lr rate")

	# general network params
	parser.add_argument("--in_channels", type=int, default=6, help="number of input channels")
	parser.add_argument("--kernal_size", type=int, default=3, help="convolutional kernal size")
	parser.add_argument("--num_channels", type=int, default=256, help="number of channels in the res blocks")
	parser.add_argument("--num_res_blocks", type=int, default=8, help="number of residual blocks")

	# value network params
	parser.add_argument("--value_accuracy_boundry", type=float, default=0.1,
	 help="threshold before value prediction is considered correct")
   
	# dataset params
	parser.add_argument("--num_games", type=int, default=4900)
	parser.add_argument("--data_split", type=list, default=[0.95, 0.04, 0.01], help="train, validation, test split")
	parser.add_argument("--data_path", type=str, default="data/big_20k_tensor/", help="path to dataset")
					 
	# checkpoint path
	
	parser.add_argument("--checkpoint_path", type=str, default="epoch=79-step=1640800.ckpt")              
	args = parser.parse_args()

	play(args)