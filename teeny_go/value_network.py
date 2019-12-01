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

class ValueNetwork(torch.nn.Module):

    def __init__(self, alpha, num_res=3, num_channel=3):
        super(ValueNetwork, self).__init__()

        self.num_res = num_res
        self.num_channel = num_channel
        self.res_block = {}

        self.define_network()
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def define_network(self):

        self.conv = torch.nn.Conv2d(self.num_channel, 1, kernel_size=1)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(self.num_channel*9*9, self.num_channel)
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.num_channel, 1)
        self.tanh = torch.nn.Tanh()

        for i in range(1, self.num_res+1):
            self.res_block["r"+str(i)] = Block(self.num_channel)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)


        out = self.conv(x)
        out = self.batch_norm(x)
        out = self.relu1(x)
        out = out.reshape(-1, self.num_channel*9*9)
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.fc2(out)
        out = self.tanh(out)

        pass

    def optimize(self, x, y, batch_size=10, iterations=10, alpha=1):

        for iter in range(iterations):
            for i in tqdm(range(num_batch)):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def main():

    vn = ValueNetwork(alpha=0.01)

    

if __name__ == "__main__":
    main()
