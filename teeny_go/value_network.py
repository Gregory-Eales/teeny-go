import torch
from tqdm import tqdm

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

class ValueNetwork(torch.nn.Module):

    def __init__(self, alpha=0.01, num_res=3, num_channel=32):
        super(ValueNetwork, self).__init__()

        self.num_res = num_res
        self.num_channel = num_channel
        self.input_channels = 11
        self.res_block = {}
        self.historical_loss = []

        self.define_network()
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def define_network(self):

        # main network
        self.pad = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(self.input_channels, self.num_channel, kernel_size=3)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)

        # value network
        self.value_conv = torch.nn.Conv2d(self.num_channel, 1, kernel_size=1)
        self.relu = torch.nn.LeakyReLU()
        self.value_batch_norm = torch.nn.BatchNorm2d(1)
        self.fc1 = torch.nn.Linear(81, 81)
        self.fc2 = torch.nn.Linear(81, 1)
        self.tanh = torch.nn.Tanh()

        for i in range(1, self.num_res+1):
            self.res_block["r"+str(i)] = Block(self.num_channel)

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)

        out = self.pad(out)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)


        for i in range(1, self.num_res+1):
            out = self.res_block["r"+str(i)].forward(out)

        # value output
        out = self.value_conv(out)
        out = self.value_batch_norm(out)
        out = self.relu(out)
        out = out.reshape(-1, 81)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out.to(torch.device('cpu:0'))
        return out

    def optimize(self, x, y, batch_size=16, iterations=10, alpha=1):



        num_batch = x.shape[0]//batch_size
        remainder = x.shape[0]%batch_size

        for iter in tqdm(range(iterations)):
            for i in range(num_batch):

                prediction = self.forward(x[i*batch_size:(i+1)*batch_size])
                loss = self.loss(prediction, y[i*batch_size:(i+1)*batch_size])
                self.historical_loss.append(loss)
                self.optimizer.zero_grad()
                if iter == 0 and i == 0: loss.backward(retain_graph=True)
                else: loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        """
def main():

    x = torch.randn(5, 11, 9, 9)

    vn = ValueNetwork(alpha=0.01)
    print(type(vn.forward(x)))



if __name__ == "__main__":
    main()
