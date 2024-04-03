import torch

import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace
import random

# import pytorch_lightning as pl

from pytorch_lightning import _logger as log
import pytorch_lightning as pl


class Block(torch.nn.Module):

    def __init__(self, kernel_size=3, num_channels=256):
        
        super(Block, self).__init__()

        self.kernel_size = kernel_size
        self.num_channel = num_channels

        self.conv1 = torch.nn.Conv2d(self.num_channel, self.num_channel, kernel_size=self.kernel_size)
        self.conv2 = torch.nn.Conv2d(self.num_channel, self.num_channel, kernel_size=self.kernel_size)
        
        self.pad = torch.nn.ZeroPad2d(1)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):

        out = self.pad(x)
        out = self.conv1(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.batch_norm(out)
        out = out + x

        out = self.relu(out)

        return out


class ValueHead(torch.nn.Module):

    def __init__(self, num_channels=256):
        super(ValueHead, self).__init__()
        
        self.conv = torch.nn.Conv2d(num_channels, 1, kernel_size=1)
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
        out = out.view(-1, 1*9*9)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out

class PolicyHead(torch.nn.Module):

    def __init__(self, num_channels=256):
        super(PolicyHead, self).__init__()

        self.conv = torch.nn.Conv2d(num_channels, 2, kernel_size=1)
        self.fc = torch.nn.Linear(2*9*9, 82)

        self.batch_norm = torch.nn.BatchNorm2d(2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)


        out = out.view(-1, 2*9*9)
        out = self.fc(out)
        return out #self.softmax(out)

    def logits(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        out = out.reshape(-1, 2*9*9)
        out = self.fc(out)
        return out

class Model(torch.nn.Module):

    # convolutional network
    # outputs 81 positions, 1 pass, 1 win/lose rating
    # residual network

    def __init__(self, lr=0.01, res_blocks = 12, num_channels = 256, in_channels = 3):

        # inherit class
        super().__init__()
        
        self.lr = lr
        
        self.internal_epoch = 0

        # define network
        self.num_res = res_blocks
        self.num_channels = num_channels
        self.input_channels = in_channels
        self.res_block = torch.nn.ModuleDict()


        self.define_network()

        #self.optimizer = torch.optim.Adam(lr=self.params.lr, params=self.parameters(), weight_decay=1e-3)
        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        # self.to(self.device)

    def define_network(self):

        # Initial Layers
        self.pad = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channels)
        self.relu = torch.nn.LeakyReLU()


        # Res Blocks
        for i in range(1, self.num_res+1):
            self.res_block["b"+str(i)] = Block()

        # Model Heads
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def forward(self, x):
        
        out = self.pad(x)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        
        for i in range(1, self.num_res+1):
            out = self.res_block["b"+str(i)](out)

        p_out = self.policy_head(out)
        v_out = self.value_head(out)

        return p_out, v_out

    def logits(self, x):
        
        out = self.pad(x)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        
        for i in range(1, self.num_res+1):
            out = self.res_block["b"+str(i)](out)

        return self.policy_head.logits(out)

def main():
    
    x = torch.randn(100, 6, 9, 9).float()
    y = torch.randn(100, 81)

    # if all elements are equal
    if torch.all(torch.eq(x.view(100, -1), x.reshape(-1, 6*9*9))):
        print("True")

    model = Model(lr=0.01, res_blocks = 1, num_channels = 256, in_channels = 6).float()

    model.forward(x)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"num params: {num_params}")

if __name__ == "__main__":
    main()
