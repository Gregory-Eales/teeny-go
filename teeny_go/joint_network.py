import torch
import pytorch_lightning as pl
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

class Block(torch.nn.Module):

    def __init__(self, num_channel, kernal_sz=2):
        super(Block, self).__init__()


        # (2*pad + 9)-kernal_sz+1 = 9

        self. kernal_size = kernal_size
        self.pad = torch.nn.ZeroPad2d(9-kernal_sz)
        self.conv1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=2)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_channel)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=2)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_channel)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.pad(x)
        print("B1:", out.shape)
        out = self.conv1(out)
        print("B2:", out.shape)
        out = self.batch_norm1(out)
        print("B3:", out.shape)
        out = self.relu1(out)
        out = self.conv2(out)
        print("B4:", out.shape)
        out = self.batch_norm2(out)
        print("B5:", out.shape)
        out = out + x
        out = self.relu2(out)
        return out


class ValueHead(torch.nn.Module):

    def __init__(self, num_channel):
        super(ValueHead, self).__init__()
        self.num_channel = num_channel
        self.conv = torch.nn.Conv2d(num_channel, 1, kernel_size=1)
        self.batch_norm = torch.nn.BatchNorm2d(num_channel)
        self.relu = torch.nn.LeakyReLU()
        self.fc1 = torch.nn.Linear(num_channel*9*9, num_channel)
        self.fc2 = torch.nn.Linear(num_channel, 1)
        self.tanh = torch.nn.Tanh()

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

    def __init__(self, num_channel):
        super(PolicyHead, self).__init__()
        self.num_channel = num_channel
        self.conv = torch.nn.Conv2d(num_channel, 2, kernel_size=1)
        self.batch_norm = torch.nn.BatchNorm2d(num_channel)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(num_channel*9*9, 82)
        self.softmax = torch.nn.Softmax()


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

class JointNetwork(torch.nn.Module):

    # convolutional network
    # outputs 81 positions, 1 pass, 1 win/lose rating
    # residual network

    def __init__(self, alpha=1e-3, input_channels=3, num_channels=128, num_res=3, is_cuda=False):

        # inherit class nn.Module
        super(JointNetwork, self).__init__()

        # define network
        self.is_cuda=is_cuda
        self.num_res = num_res
        self.num_channels = num_channels
        self.input_channels = input_channels
        self.res_block = torch.nn.ModuleDict()


        self.define_network()
      
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.loss = torch.nn.BCELoss()
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
            self.res_block["b"+str(i)] = Block(num_channel=self.num_channels)

        # Model Heads
        self.value_head = ValueHead(self.num_channels)
        self.policy_head = PolicyHead(self.num_channels)


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


    

def main():
    
    x = torch.randn(100, 3, 9, 9)
    y = torch.randn(100, 81)

    jn = JointNetwork()

    jn.forward(x)

if __name__ == "__main__":
    main()
