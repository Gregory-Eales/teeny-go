import torch



class Block(torch.nn.Module):

    def __init__(self, num_channel)
        super(Block, self).__init__()
        # convolution
        self.conv = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        # batch normalize
        self.batch_norm = torch.nn.BatchNorm2d(num_channel)
        # relu
        self.relu = torch.nn.ReLU()

        self.downsample = downsample

    def forward(self, x):

        identity = x

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class TeenyGoNetwork(torch.nn.Module):

    # convolutional network
    # outputs 81 positions, 1 pass, 1 win/lose rating
    # residual network

    def __init__(self, input_channels=11,num_channels=256, num_res_blocks=20):

        # inherit class nn.Module
        super(TeenyGoNetwork, self).__init__()

        # define network
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.input_channels = input_channels
        self.layers = {}
        self.optimizer = None

        # initilize network
        self.initialize_layers()
        self.initialize_optimizer()


    def predict(self):
        pass

    def initialize_layers(self):

        self.l["l1"] = torch.nn.Conv2d(self.input_channels, self.num_channels)

        for i in range(2, self.num_res_blocks+1):
            self.layers["l"+str(i)] = Block()

    def initialize_optimizer(self):
        pass



def main():
    x = torch.randn(100, 9, 9, 20)
    tgn = TeenyGoNetwork()
    tgn.predict(x)

if __name__ == "__main__":
    main()
