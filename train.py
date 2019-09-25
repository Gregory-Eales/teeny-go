from utils.trainer import Trainer
from teeny_go.teeny_go_network import TeenyGoNetwork
from matplotlib import pyplot as plt

import time

import torch

torch.cuda.empty_cache()


tgn = TeenyGoNetwork(num_channels=64, num_res_blocks=5, is_cuda=True)
tgn.load_state_dict(torch.load("models/Model-R5-C64/Model-R5-C64-Vtrained.pt"))
tgn.cuda()

"""
trainer = Trainer(network=tgn)
trainer.train_self_play(num_games=500, is_cuda=True, iterations=250)
"""

x = []
y = []

path = "data/aya_dataset/"

for i in range(40000):
    x.append(torch.load("{}DataX{}{}".format(path, i, ".pt")))
    y.append(torch.load("{}DataY{}{}".format(path, i, ".pt")))

x = torch.cat(x, 0)
y = torch.cat(y, 0)
#x = x.cuda().type(torch.cuda.FloatTensor)
#y = y.cuda().type(torch.cuda.FloatTensor)

hist = []
plt.ion()

for i in range(1):
    for j in range(1):

        tgn.optimize(x, y, batch_size=5000, iterations=10, alpha=0.0001)
        torch.cuda.empty_cache()
        hist.append(tgn.hist_cost[-1].tolist())
        print(hist[-1])
        plt.plot(hist)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

path = "models/Model-R{}-C{}/".format(5, 64)
filename = "Model-R{}-C{}-V{}.pt".format(5, 64, "trained")
torch.save(tgn.state_dict(), path+filename)
