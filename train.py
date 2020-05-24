import torch
import pytorch_lightning as pl
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace


class Block(torch.nn.Module):

	def __init__(self, hparams):
		
		super(Block, self).__init__()

		self.hparams = hparams
	
		self.kernal_size = self.hparams.kernal_size
		self.num_channel = self.hparams.num_channels

		self.conv1 = torch.nn.Conv2d(self.num_channel, self.num_channel, kernel_size=self.kernal_size)
		self.conv2 = torch.nn.Conv2d(self.num_channel, self.num_channel, kernel_size=self.kernal_size)
		
		self.pad = torch.nn.ZeroPad2d(1)
		self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
		self.relu = torch.nn.ReLU()

	def forward(self, x):

		out = self.pad(x)
		out = self.conv1(out)
		out = self.batch_norm(out)
		out = self.relu(out)

		out = self.pad(x)
		out = self.conv2(out)
		out = self.batch_norm(out)
		out = out + x

		out = self.relu(out)

		return out


class ValueHead(torch.nn.Module):

	def __init__(self, hparams):
		super(ValueHead, self).__init__()
		
		self.hparams = hparams
		self.num_channel = self.hparams.num_channels

		self.conv = torch.nn.Conv2d(self.num_channel, 1, kernel_size=1)
		self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
		self.fc1 = torch.nn.Linear(self.num_channel*9*9, self.num_channel)
		self.fc2 = torch.nn.Linear(self.num_channel, 1)

		self.tanh = torch.nn.Tanh()
		self.relu = torch.nn.LeakyReLU()

	def forward(self, x):

		out = self.conv(out)
		out = self.batch_norm(out)
		out = self.relu(out)
		shape = out.shape
		out = out.reshape(-1, self.num_channel*9*9)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.tanh(out)
		return out

class PolicyHead(torch.nn.Module):

	def __init__(self, hparams):
		super(PolicyHead, self).__init__()


		self.hparams = hparams
	
		self.kernal_size = self.hparams.kernal_size
		self.num_channel = self.hparams.num_channels

		self.conv = torch.nn.Conv2d(self.num_channel, 2, kernel_size=1)
		self.fc = torch.nn.Linear(self.num_channel*9*9, 82)

		self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
		self.softmax = torch.nn.Softmax()
		self.relu = torch.nn.LeakyReLU()


	def forward(self, x):

		out = self.conv(x)
		print("P1:", out.shape)
		out = self.batch_norm(out)
		print("P2:", out.shape)
		out = self.relu(out)
		out = out.reshape(-1, self.num_channel*9*9)
		print("P3:", out.shape)
		out = self.fc(out)
		print("P4:", out.shape)
		out = self.softmax(out)
		return out

class JointNetwork(pl.LightningModule):

	# convolutional network
	# outputs 81 positions, 1 pass, 1 win/lose rating
	# residual network

	def __init__(self, hparams):

		# inherit class nn.Module
		super(JointNetwork, self).__init__()

		self.hparams = hparams

		# define network
		self.num_res = self.hparams.num_res_blocks
		self.num_channels = self.hparams.num_channels
		self.input_channels = self.hparams.in_channels
		self.res_block = torch.nn.ModuleDict()


		self.define_network()
	  
		self.optimizer = torch.optim.Adam(lr=self.hparams.lr, params=self.parameters())

		self.policy_loss = torch.nn.BCELoss()
		self.value_loss = torch.nn.MSELoss()

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
		self.to(self.device)

	def define_network(self):

		# Initial Layers
		self.pad = torch.nn.ZeroPad2d(1)
		self.conv = torch.nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3)
		self.batch_norm = torch.nn.BatchNorm2d(self.num_channels)
		self.relu = torch.nn.ReLU()


		# Res Blocks
		for i in range(1, self.num_res+1):
			self.res_block["b"+str(i)] = Block(hparams=self.hparams)

		# Model Heads
		self.value_head = ValueHead(self.hparams)
		self.policy_head = PolicyHead(self.hparams)


	def forward(self, x):

		out = self.pad(x)
		out = self.conv(out)
		out = self.batch_norm(out)
		out = self.relu(out)


		for i in range(1, self.num_res+1):
			out = self.res_block["b"+str(i)](out)

		p_out = self.policy_head(out)
		v_out = self.value_head(out)

		return torch.cat([p_out, v_out], dim=1).to("cpu:0")

	def training_step(self, batch, batch_idx):
		x, y = batch
		p, v = self.forward(x)

		p_loss = self.policy_loss(p, y)
		v_loss = self.value_loss(p, y)

		tensorboard_logs = {'policy_train_loss': p_loss,
		 "value_train_loss": v_loss}


		return {'loss': loss, 'log': tensorboard_logs}

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		return {'val_loss': F.cross_entropy(y_hat, y)}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		tensorboard_logs = {'avg_val_loss': avg_loss}
		return {'val_loss': avg_loss, 'log': tensorboard_logs}

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.forward(x)
		return {'test_loss': F.cross_entropy(y_hat, y)}

	def test_epoch_end(self, outputs):

		avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

		tensorboard_logs = {'test_val_loss': avg_loss}
		return {'test_loss': avg_loss, 'log': tensorboard_logs}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


	def combine_shuffle_data(self, x, y):

		rand_perm = torch.randperm(x.shape[0])

		x = torch.cat(x).float()
		x = x[rand_perm]
		y = torch.cat(y).float()
		y = y[:,82].reshape(y.shape[0], -1)[rand_perm]

		return x, y


	def prepare_data(self):
		
		num_games = self.hparams.num_games
		num_train = self.hparams.num_train
		num_validation = self.hparams.num_validation
		num_test = self.hparams.num_test
		

		path = self.hparams.games_path

		x = []
		y = []
		x_path = path + "DataX"
		y_path = path + "DataY"

		for i in range(num_games):
			try:
				x.append(torch.load(x_path+str(i)+".pt"))
				y.append(torch.load(y_path+str(i)+".pt"))
			except:pass

		[0.9, 0.05, 0.05]

		split = self.hparams.data_split

		trn_1 = 0
		trn_2 = int(split[0]*num_games)

		val_1 = trn_2
		val_2 = trn_2 + int(split[1]*num_games)

		test_1 = val_2
		test_2 = val_2 + int(split[2]*num_games)

		x_train, y_train = self.combine_shuffle_data(x[trn_1:trn_2], y[trn_1:trn_2])
		x_val, y_val = self.combine_shuffle_data(x[val_1:val_2], y[val_1:val_2])
		x_test, y_test = self.combine_shuffle_data(x[test_1:test_2], y[test_1:test_2])

		# assign to use in dataloaders
		self.train_dataset = GoDataset(self.hparams, x_train, y_train)
		self.val_dataset = GoDataset(self.hparams, x_val, y_val)
		self.test_dataset = GoDataset(self.hparams, x_test, y_test)

	def train_dataloader(self):
		log.info('Training data loader called.')
		return DataLoader(self.train_dataset)

	def val_dataloader(self):
		log.info('Validation data loader called.')
		return DataLoader(self.val_dataset)

	def test_dataloader(self):
		log.info('Test data loader called.')
		return DataLoader(self.test_dataset)

	@staticmethod
	def add_model_specific_args(parent_parser):

		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--learning_rate', default=0.02, type=float)
		parser.add_argument('--batch_size', default=32, type=int)
		parser.add_argument('--max_nb_epochs', default=2, type=int)

		return parser
	

class GoDataset(Dataset):

	def __init__(self, hparams, x, y):

		self.hparams = hparams

		super(GoDataset, self).__init__()

	def load_data(self):

		num_games = self.hparams.num_games
		num_train = self.hparams.num_train
		num_validation = self.hparams.num_validation
		num_test = self.hparams.num_test
		

		path = self.hparams.games_path

		self.x = []
		self.y = []
		x_path = path + "DataX"
		y_path = path + "DataY"

		for i in range(num_games):
			try:
				x.append(torch.load(x_path+str(i)+".pt"))
				y.append(torch.load(y_path+str(i)+".pt"))
			except:pass
		x = torch.cat(x).float()
		rand_perm = torch.randperm(x.shape[0])
		x = x[rand_perm]
		y = torch.cat(y).float()
		y = y[:,82].reshape(y.shape[0], -1)[rand_perm]

	def __len__(self):
		return len(self.x.shape[0])
	
	def __iter__(self):
		num_batch = self.x.shape[0]//self.hparams.batch_size
		rem_batch = self.x.shape[0]%self.hparams.batch_size
		
		for i in range(num_batch):
			i1, i2 = i*16, (i+1)*self.hparams.batch_size
			yield self.x[i1:i2], self.y[i1:i2]
		
		
		i1 = -rem_batch
		i2 = 0
		yield self.x[i1:i2], self.y[i1:i2]


def main(args):
	
	net = JointNetwork(args)
	
	trainer = pl.Trainer(gpus=args.gpu,
						 max_epochs=args.max_epochs,
						 check_val_every_n_epoch = 1,
						 min_epochs = 1,
						 train_percent_check = 1.0,
						 val_percent_check = 1.0,
						 test_percent_check = 1.0,
						 val_check_interval = 1.0,
						 log_save_interval = 100,
						 row_log_interval = 10,
						 precision= 16,
						 print_nan_grads = False,
						 weights_summary = 'full',
						 weights_save_path = None,
						 num_sanity_val_steps = 2,
						 resume_from_checkpoint = None,
						 auto_lr_find = False,
						 use_amp=False)
	
	#trainer.fit(net)

if __name__ == '__main__':


	torch.manual_seed(0)
	np.random.seed(0)

	parser = ArgumentParser()

	# training params
	parser.add_argument("--gpu", type=int, default=0, help="number of gpus")
	parser.add_argument("--early_stopping", type=bool, default=True, help="whether early stopping is used or not")
	parser.add_argument("--max_epochs", type=int, default=200, help="number of gpus")
	parser.add_argument("--batch_size", type=int, default=64, help="size of training batch")
	parser.add_argument("--lr", type=int, default=1e-2, help="learning rate")
	parser.add_argument("--accumulate_grad_batches", type=int, default=64, help="grad batches")
	parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="")
   
	# dataset params
	parser.add_argument("--num_games", type=int, default=1028, help="number of games to load")
	parser.add_argument("--data_split", type=list, default=[0.9, 0.05, 0.05], help="test, train, and validation split")
	parser.add_argument("--data_path", type=str, default="/input/godataset/new_ogs_tensor_games/",
					   help="path to data")

	# general network params
	parser.add_argument("--in_channels", type=int, default=3, help="")
	parser.add_argument("--kernal_size", type=int, default=3, help="")
	parser.add_argument("--num_channels", type=int, default=128, help="number of channels in the res blocks")
	parser.add_argument("--num_res_blocks", type=int, default=3, help="number of residual blocks")


	# value network params
	parser.add_argument("--value_accuracy_boundry", type=float, default=0.3,
	 help="threshold before value prediction is considered correct")


	# run
	args = parser.parse_args()
	main(args)