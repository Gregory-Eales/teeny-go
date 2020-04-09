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

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

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

    def __init__(self, alpha, num_res=3, num_channel=3):
        super(PolicyNetwork, self).__init__()

        #self.input_channels = num_channel
        self.state_channel = 11
        self.num_res = num_res
        self.res_block = torch.nn.ModuleDict()
        self.num_channel = num_channel

        self.define_network()
        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def define_network(self):
        #policy head
        self.policy_conv = torch.nn.Conv2d(self.num_channel, 2, kernel_size=1)
        self.policy_batch_norm = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
        self.policy_fc = torch.nn.Linear(2*9*9, 82)
        self.softmax = torch.nn.Softmax(dim=1)

        # network start
        self.pad = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(self.state_channel, self.num_channel, kernel_size=3)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
        self.relu = torch.nn.ReLU()

        for i in range(1, self.num_res+1):
            self.res_block["r"+str(i)] = Block(self.num_channel)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)

        out = self.pad(out)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        for i in range(1, self.num_res+1):
            out = self.res_block["r"+str(i)](out)

        # policy head
        out = self.policy_conv(out)
        out = self.policy_batch_norm(out)
        out = self.relu(out)
        out = out.reshape(-1, 2*9*9)
        out = self.policy_fc(out)
        out = self.softmax(out)
        return out


    def optimize(self, x, y, x_t=None, y_t=None, iter=10, batch_sz=64):


        num_batchs = x.shape[0]//batch_sz
        remainder = x.shape[0]%batch_sz

        for i in range(iter):
            for n in range(num_batchs):


                pass

def main():
    pn = PolicyNetwork(alpha=0.01, num_res=3, num_channel=64)

    x = torch.ones(100, 11, 9, 9)
    print(pn.forward(x).shape)

if __name__ == "__main__": main()
