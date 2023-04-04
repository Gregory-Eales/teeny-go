import torch
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace
import random


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
        self.relu = torch.nn.LeakyReLU()

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
        self.batch_norm = torch.nn.BatchNorm2d(1)
        self.fc1 = torch.nn.Linear(9*9, 64)
        self.fc2 = torch.nn.Linear(64, 1)

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out = x

        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = out.reshape(-1, 1*9*9)
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
        self.fc = torch.nn.Linear(2*9*9, 82)

        self.batch_norm = torch.nn.BatchNorm2d(2)
        self.softmax = torch.nn.Softmax()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        out = out.reshape(-1, 2*9*9)
        out = self.fc(out)
        out = self.softmax(out)
        return out

class JointNetwork(pl.LightningModule):

    # convolutional network
    # outputs 81 positions, 1 pass, 1 win/lose rating
    # residual network

    def __init__(self, hparams):

        # inherit class
        super().__init__()

        self.hparams = hparams
        
        self.internal_epoch = 0

        # define network
        self.num_res = self.hparams.num_res_blocks
        self.num_channels = self.hparams.num_channels
        self.input_channels = self.hparams.in_channels
        self.res_block = torch.nn.ModuleDict()


        self.define_network()

        #self.prepare_data()
      
        #self.optimizer = torch.optim.Adam(lr=self.hparams.lr, params=self.parameters(), weight_decay=1e-3)

        self.value_loss = torch.nn.MSELoss()

        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        #self.to(self.device)

    def define_network(self):

        # Initial Layers
        self.pad = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channels)
        self.relu = torch.nn.LeakyReLU()


        # Res Blocks
        for i in range(1, self.num_res+1):
            self.res_block["b"+str(i)] = Block(hparams=self.hparams)

        # Model Heads
        self.value_head = ValueHead(self.hparams)
        self.policy_head = PolicyHead(self.hparams)

    def policy_loss(self, p, a):

        action_probabilities = torch.distributions.Categorical(p)

        log_prob = action_probabilities.log_prob(a.max(dim=1)[1])

        return -log_prob.mean()


    def forward(self, x):
        
       
        out = self.pad(x)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        
        for i in range(1, self.num_res+1):
            out = self.res_block["b"+str(i)](out)

        p_out = self.policy_head(out)
        v_out = self.value_head(out)

        return p_out, v_out # ], dim=1).to("cpu:0")

    def training_step(self, batch, batch_idx):
        x, y = batch
        p, v = self.forward(x)

        p_loss = self.policy_loss(p, y[:,0:82].reshape(-1, 82))#*1.5e1
        v_loss = self.value_loss(v, y[:,82].reshape(-1, 1))
        
        loss = p_loss + v_loss

        tensorboard_logs = {'policy_train_loss': p_loss,
         "value_train_loss": v_loss}

        return {'loss': loss, 'log': tensorboard_logs}
    
    
    def get_policy_accuracy(self, p, y):

        p = p.cpu()
        y = y.cpu()

        c = torch.zeros(y.shape[0], y.shape[1])
        
        c[p == p.max(dim=0)[0]] = 1
        c[p != p.max(dim=0)[0]] = 0

        correct_percent = torch.sum(c*y) / y.shape[0]

        return correct_percent
    
    def get_value_accuracy(self, v, y):

        v.cpu()
        y.cpu()

        c = torch.zeros(y.shape[0], y.shape[1])
        k = torch.zeros(y.shape[0], y.shape[1])
        
        
        c[v<-1*self.hparams.value_accuracy_boundry] = -1
        c[v>self.hparams.value_accuracy_boundry] = 1
        
        k[y<0] = -1
        k[y>0] = 1

        correct_percent = torch.sum(((c+k)/2)**2) / y.shape[0]

        return correct_percent

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p, v = self.forward(x)

        p_loss = self.policy_loss(p, y[:,0:82].reshape(-1, 82))

        v_loss = self.value_loss(v, y[:,82].reshape(-1, 1))

        
        p_acc = self.get_policy_accuracy(p, y[:,0:82].reshape(-1, 82))
        v_acc = self.get_value_accuracy(v, y[:,82].reshape(-1, 1))
        
        loss = p_loss + v_loss
        
        tensorboard_logs = {'policy_val_loss': p_loss, "value_val_loss": v_loss,
                           "value_val_accuracy":v_acc, "policy_val_accuracy":p_acc}

        return {'val_loss': loss, "value_val_accuracy":v_acc, "policy_val_accuracy":p_acc,
                'policy_val_loss': p_loss, "value_val_loss": v_loss, 'log':tensorboard_logs}

    def validation_epoch_end(self, outputs):
        
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        avg_value_loss = torch.stack([x["value_val_loss"] for x in outputs]).mean()
        avg_policy_loss = torch.stack([x["policy_val_loss"] for x in outputs]).mean()
        
        avg_value_accuracy = torch.stack([x["value_val_accuracy"] for x in outputs]).mean()
        avg_policy_accuracy = torch.stack([x["policy_val_accuracy"] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss':avg_val_loss, "policy_val_accuracy":avg_policy_accuracy,
                            "value_val_accuracy":avg_value_accuracy, "policy_val_loss":avg_policy_loss,
                            "value_val_loss":avg_value_loss}
        
        
        
        
        if self.internal_epoch % 10 == 0:
            torch.save(self.state_dict(), "models/joint_model_v{}.pt".format(self.internal_epoch))
        
        self.internal_epoch += 1

        return {'avg_val_loss': avg_val_loss, 'log':tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_val_loss': avg_loss}
        
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def combine_shuffle_data(self, x, y):

        

        x = torch.cat(x).float()
        y = torch.cat(y).float()

        rand_perm = torch.randperm(x.shape[0])

        x = x[rand_perm]
        y = y[rand_perm]

        return x, y


    def prepare_data(self):
        
        num_games = self.hparams.num_games
        path = self.hparams.data_path

        x = []
        y = []
        x_path = path + "DataX"
        y_path = path + "DataY"

        for i in range(num_games):
            
            try:
                
                
                pre_x = torch.load(x_path+str(i)+".pt").float()
                
                pre_y = torch.load(y_path+str(i)+".pt").float()
                #pre_y[:,82] *= torch.tanh(torch.linspace(0, 1.5, steps=pre_y.shape[0]))
                
                #rand_perm = torch.randperm(pre_x.shape[0])
                
                x.append(pre_x)
                y.append(pre_y)
                
            except:pass

        split = self.hparams.data_split

        trn_1 = 0
        trn_2 = int(split[0]*len(x))

        val_1 = trn_2
        val_2 = trn_2 + int(split[1]*len(x))

        test_1 = val_2
        test_2 = val_2 + int(split[2]*len(x))
        
        """
        print(len(x))
        print(trn_1, trn_2)
        print(val_1, val_2)
        print(test_1, test_2)
        """

        x_train, y_train = self.combine_shuffle_data(x[trn_1:trn_2], y[trn_1:trn_2])
        
        print("Loaded Training Data")
        
        x_val, y_val = self.combine_shuffle_data(x[val_1:val_2], y[val_1:val_2])
        print("Loaded Validation Data")
        x_test, y_test = self.combine_shuffle_data(x[test_1:test_2], y[test_1:test_2])
        print("Loaded Test Data")

        print(x_train.shape)
        print(y_train.shape)
       

        # assign to use in dataloaders
        self.train_dataset = GoDataset(self.hparams, x_train, y_train)
        self.val_dataset = GoDataset(self.hparams, x_val, y_val)
        self.test_dataset = GoDataset(self.hparams, x_test, y_test)

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--max_nb_epochs', default=2, type=int)
        """
        return parent_parser
    

class GoDataset(Dataset):

    def __init__(self, hparams, x, y):

        self.hparams = hparams

        super(GoDataset, self).__init__()

        self.x = x
        self.y = y

    def load_data(self):

        """
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

        """

        pass

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __iter__(self):
        num_batch = self.x.shape[0]//self.hparams.batch_size
        rem_batch = self.x.shape[0]%self.hparams.batch_size
        
        for i in range(num_batch):
            i1, i2 = i*self.hparams.batch_size, (i+1)*self.hparams.batch_size
            yield self.x[i1:i2], self.y[i1:i2]
        
        
        i1 = -rem_batch
        i2 = 0
        yield self.x[i1:i2], self.y[i1:i2]

def main():
    
    x = torch.randn(100, 3, 9, 9)
    y = torch.randn(100, 81)

    jn = JointNetwork(None)

    jn.forward(x)

if __name__ == "__main__":
    main()
