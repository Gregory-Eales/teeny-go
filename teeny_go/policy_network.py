import torch

class Block(torch.nn.Module):

    def __init__(self, num_channel):
        super(Block, self).__init__()
        self.pad1 = torch.nn.ZeroPad2d(1)
        self.conv1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_channel)
        self.relu1 = torch.nn.ReLU()
        self.pad2 = torch.nn.ZeroPad2d(1)
        self.conv2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_channel)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = out + x
        out = self.relu2(out)
        return out

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, num_res=3):
        super(PolicyNetwork, self).__init__()

        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)
        pass

    def optimize(self):
        pass
