import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt
"""
torch.cuda.empty_cache()
value_net = ValueNetwork(alpha=0.001, num_res=5, num_channel=64)
print(value_net.device)
x = []
y = []
x_path = "data/pro_game_dataset/DataX"
y_path = "data/pro_game_dataset/DataY"
for i in range(10):
    try:
        x.append(torch.load(x_path+str(i)+".pt"))
        y.append(torch.load(y_path+str(i)+".pt"))
    except:pass
x = torch.cat(x).float()
rand_perm = torch.randperm(x.shape[0])
x = x[rand_perm]
y = torch.cat(y).float()
y = y[:,82].reshape(y.shape[0], -1)[rand_perm]
print(y.shape)
print(x.shape)
value_net.optimize(x, y, batch_size=64, iterations=3, alpha=0.01)
plt.plot(value_net.historical_loss)
plt.show()
torch.save(value_net.state_dict(), "models/value-net/VN-R3-C128-V1.pt")
# process dataset
"""


import os
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import torchvision.transforms as transforms
import gym
import torch
import numpy as np

from teeny_go import joint_network
from utils import GoDataset

def main(args):
	
	j_net = JointNetwork(args)
	

if __name__ == '__main__':


	torch.manual_seed(0)
	np.random.seed(0)

	parser = ArgumentParser()

	# training params
	parser.add_argument("--gpu", type=int, default=0, help="number of gpus")
	parser.add_argument("--num_epochs", type=int, default=200, help="number of gpus")
	parser.add_argument("--batch_size", type=int, default=64, help="size of training batch")
	parser.add_argument("--lr", type=int, default=1e-2, help="learning rate")
	parser.add_argument("--accumulate_grad_batches", type=int, default=64, help="grad batches")

	# general network params
	parser.add_argument("--in_channel", type=int, default=3, help="")
	parser.add_argument("--kernal_size", type=int, default=3, help="")
	parser.add_argument("--num_channels", type=int, default=128, help="number of channels in the res blocks")
	parser.add_argument("--num_res_blocks", type=int, default=3, help="number of residual blocks")

	# value network params
	parser.add_argument("--value_accuracy_boundry", type=float, default=0.3,
	 help="threshold before value prediction is considered correct")

	# residual block params


	# run
	args = parser.parse_args()
	main(args)