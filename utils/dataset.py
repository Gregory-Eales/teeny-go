import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class GoDataSet(Dataset):

	def __init__(self, path, num_games=100):


		self.num_games = num_games
		self.path = path + "Data{}{}.pt"
		self.x, self.y = self.load_data()
	
	def load_data(self):
		x = []
		y = []

		try:
			for i in range(num_games):
			    x.append(torch.load(self.path.format("X", i)))
			    y.append(torch.load(self.path.format("Y", i)))
	    except:
	    	print("Unable to Load Data")

		x = torch.cat(x).float()
		y = torch.cat(y).float()

		return x, y

	def __len__(self):
        return len(self.num_games)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]