import torch



class Block(torch.nn.Module):

    def __init__(self, num_channel):
        super(Block, self).__init__()
        self.pad1 = torch.nn.ZeroPad2d(1)
        # convolution
        self.conv1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        # batch normalize
        self.batch_norm1 = torch.nn.BatchNorm2d(num_channel)
        # relu
        self.relu1 = torch.nn.ReLU()

        self.pad2 = torch.nn.ZeroPad2d(1)
        # convolution
        self.conv2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        # batch normalize
        self.batch_norm2 = torch.nn.BatchNorm2d(num_channel)
        # relu
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

class TeenyGoNetwork(torch.nn.Module):

    # convolutional network
    # outputs 81 positions, 1 pass, 1 win/lose rating
    # residual network

    def __init__(self, input_channels=11, num_channels=256, num_res_blocks=5):

        # inherit class nn.Module
        super(TeenyGoNetwork, self).__init__()

        # define network
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.input_channels = input_channels
        self.res_layers = {}
        self.optimizer = None

        # initilize network
        self.pad = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channels)
        self.relu = torch.nn.ReLU()

        self.initialize_layers()
        self.initialize_optimizer()


    def predict(self, x):

        out = self.pad(x)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        for i in range(1, self.num_res_blocks+1):
            out = self.res_layers["l"+str(i)](out)

        return out


    def initialize_layers(self):


        for i in range(1, self.num_res_blocks+1):
            self.res_layers["l"+str(i)] = Block(self.num_channels)

    def initialize_optimizer(self):
        pass



def main():
    x = torch.randn(100, 11, 9, 9)
    tgn = TeenyGoNetwork(num_res_blocks=10)
    tgn.predict(x)
    

if __name__ == "__main__":
    main()
