from torch.utils.data import Dataset



class GoDataset(Dataset):

	def __init__(self, hparams):

		self.hparams = hparams

		super(GoDataset, self).__init__()

		self.load_data()

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