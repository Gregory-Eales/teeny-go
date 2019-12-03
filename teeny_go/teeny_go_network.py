import torch
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

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


class ValueHead(torch.nn.Module):

    def __init__(self, num_channel):
        super(ValueHead, self).__init__()
        self.num_channel = num_channel
        self.conv = torch.nn.Conv2d(num_channel, 1, kernel_size=1)
        self.batch_norm = torch.nn.BatchNorm2d(num_channel)
        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(num_channel*9*9, num_channel)
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(num_channel, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):

        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu1(out)
        shape = out.shape
        out = out.reshape(-1, self.num_channel*9*9)
        out = self.fc1(out)
        out = self.relu2(out)
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
        self.softmax = torch.nn.Sigmoid()


    def forward(self, x):

        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = out.reshape(-1, self.num_channel*9*9)
        out = self.fc(out)
        out = self.softmax(out)
        return out

class TeenyGoNetwork(torch.nn.Module):

    # convolutional network
    # outputs 81 positions, 1 pass, 1 win/lose rating
    # residual network

    def __init__(self, input_channels=11, num_channels=256, num_res_blocks=5, is_cuda=False):

        # inherit class nn.Module
        super(TeenyGoNetwork, self).__init__()

        # define network
        self.is_cuda=is_cuda
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.input_channels = input_channels
        self.res_layers = {}
        self.optimizer = None

        # initilize network

        if self.is_cuda:
            self.pad = torch.nn.ZeroPad2d(1).cuda()
            self.conv = torch.nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3).cuda()
            self.batch_norm = torch.nn.BatchNorm2d(self.num_channels).cuda()
            self.relu = torch.nn.ReLU().cuda()
            self.value_head = ValueHead(self.num_channels).cuda()
            self.policy_head = PolicyHead(self.num_channels).cuda()
        else:
            self.pad = torch.nn.ZeroPad2d(1)
            self.conv = torch.nn.Conv2d(self.input_channels, self.num_channels, kernel_size=3)
            self.batch_norm = torch.nn.BatchNorm2d(self.num_channels)
            self.relu = torch.nn.ReLU()
            self.value_head = ValueHead(self.num_channels)
            self.policy_head = PolicyHead(self.num_channels)

        self.initialize_layers()
        self.initialize_optimizer()

        self.hist_cost = []

    def forward(self, x):

        out = self.pad(x)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)


        for i in range(1, self.num_res_blocks+1):
            out = self.res_layers["l"+str(i)](out)

        policy_out = self.policy_head(out)
        value_out = self.value_head(out)

        return torch.cat((policy_out, value_out), 1)


    def initialize_layers(self):
        if self.is_cuda:
            for i in range(1, self.num_res_blocks+1):
                self.res_layers["l"+str(i)] = Block(self.num_channels).cuda()
        else:
            for i in range(1, self.num_res_blocks+1):
                self.res_layers["l"+str(i)] = Block(self.num_channels)

    def initialize_optimizer(self, learning_rate=0.00001):
        #Loss function

        #Optimizer
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def loss(self, prediction, y, alpha):

        policy, value = prediction[:,0:82], prediction[:,82]
        y_policy, outcome = y[:,0:82], y[:,82]
        """
        print(value - outcome)
        print("#############")
        print(torch.log(policy))
        """
        #(value - outcome)**2)[:, None]
        #loss = (torch.sum(0.5*((value - outcome)**2)) - torch.sum(0.5*outcome[:, None]*torch.log(policy+1)))/prediction.shape[0]
        #loss = alpha*(torch.sum(0.5*((y - prediction)**2)))
        epsilon = 0.0000000001
        loss = (torch.sum(0.5*((value - outcome)**2)) + 2*torch.sum(-y_policy*torch.log(policy+epsilon) - 2*(1-y_policy)*torch.log(1-policy+epsilon)))/prediction.shape[0]
        return loss

    def mean_square(self, prediction, y, alpha):

        policy, value = prediction[:,0:82], prediction[:,82]
        y_policy, outcome = y[:,0:82], y[:,82]
        #(value - outcome)**2)[:, None]
        #loss = (torch.sum(0.5*((value - outcome)**2)) - torch.sum(0.5*outcome[:, None]*torch.log(policy+1)))/prediction.shape[0]
        error = alpha*(torch.sum(0.5*((y - prediction)**2)))
        #loss = (torch.sum(0.5*((value - outcome)**2)) - torch.sum(y_policy*torch.log(policy+0.00000001) - (1-y_policy)*torch.log(1-policy+0.00000001)))/prediction.shape[0]
        return error

    def accuracy_metric(self, prediction, y):
        #print("#####################################")
        #print(prediction[35])
        #print(y[35])
        #print("#####################################")
        diff = torch.abs(prediction[:,0:82]-y[:,0:82])
        diff = 100*(1 - torch.sum(diff / (diff.shape[0]*84)))
        return diff


    def optimize(self, x, y, batch_size=10, iterations=10, alpha=1):

        num_batch = x.shape[0]//batch_size
        remainder = x.shape[0]%batch_size
        plt.ion()
        for iter in range(iterations):
            for i in tqdm(range(num_batch)):
                self.optimizer.zero_grad()

                x_batch = x[i*batch_size:(i+1)*batch_size]
                y_batch = y[i*batch_size:(i+1)*batch_size]

                output = self.forward(x_batch.cuda().type(torch.cuda.FloatTensor))
                loss = self.loss(output, y_batch.cuda().type(torch.cuda.FloatTensor), alpha)
                error = self.accuracy_metric(output, y_batch.cuda().type(torch.cuda.FloatTensor))
                #print(error)

                l = error.clone()
                self.hist_cost.append(l.tolist())
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
                plot_nums = np.array(self.hist_cost)
                plt.title("Accuracy per Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy(%)")
                plt.plot(plot_nums, label="Accuracy: {}%".format(round(self.hist_cost[-1], 3)))#/plot_nums.max())
                plt.legend(loc="lower right")
                plt.draw()
                plt.pause(0.0001)
                plt.clf()

            if iter%20 == 0:
                alpha*=0.8

            self.optimizer.zero_grad()
            output = self.forward(x[-remainder:-1].cuda().type(torch.cuda.FloatTensor))
            loss = self.loss(output, y[-remainder:-1].cuda().type(torch.cuda.FloatTensor), alpha)
            #self.hist_cost.append(loss)
            loss.backward()
            self.optimizer.step()

            path = "models/Model-R{}-C{}/".format(self.num_res_blocks, self.num_channels)
            filename = "Model-R{}-C{}-V{}.pt".format(self.num_res_blocks, self.num_channels, "SL"+str(iter))
            torch.save(self.state_dict(), path+filename)

        return self.hist_cost

def main():
    x = torch.abs(torch.randn(1000, 11, 9, 9))
    #y = torch.abs(torch.randn(100, 83))
    tgn = TeenyGoNetwork(num_res_blocks=5, num_channels=96)
    t = time.time()
    pred = tgn(x)
    print("prediction time:", round(time.time()-t, 5), "s")




if __name__ == "__main__":
    main()
