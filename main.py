#from teeny_go.go_trainer import GoTrainer
#import logging
import torch

from teeny_go.teeny_go_network import TeenyGoNetwork
from matplotlib import pyplot as plt

torch.cuda.init()

def load_data(n):
    x_data = []
    y_data = []

    for i in range(n*200 + 1, (n+1)*200):
        x_data.append(torch.load("data\Model_R5_C64_DataX"+str(i)+".pt"))
        y_data.append(torch.load("data\Model_R5_C64_DataY"+str(i)+".pt"))

    x_data = torch.cat(x_data)
    y_data = torch.cat(y_data)

    print(x_data.shape)
    print(y_data.shape)

    return x_data, y_data


cost_list = []

for j in range(10):
    for i in range(10):

        x, y = load_data(i)
        tgn = TeenyGoNetwork(num_channels=32, num_res_blocks=3)
        tgn.load_state_dict(torch.load("Models\Model_R3_C32_V0.pt"))

        tgn.cuda()

        cost = tgn.optimize(x.cuda(), y.cuda(), batch_size=x.shape[0], iterations=10)
        cost_list+=cost
        torch.save(tgn.state_dict(), "Models\Model_R3_C32_V0.pt")
        torch.cuda.empty_cache()

plt.plot(cost_list)
plt.show()
