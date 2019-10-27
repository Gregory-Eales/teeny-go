from utils.trainer import Trainer
from teeny_go.teeny_go_network import TeenyGoNetwork
from matplotlib import pyplot as plt

import time
import torch

tgn = TeenyGoNetwork(num_channels=256, num_res_blocks=8, is_cuda=True)

"""
SELF PLAY LOOP

trainer = Trainer(network=tgn)
trainer.train_self_play(num_games=500, is_cuda=True, iterations=250)
"""

x = []
y = []
path = "data/aya_dataset/"

for i in range(10):
    x.append(torch.load("{}DataX{}{}".format(path, i, ".pt")))
    y.append(torch.load("{}DataY{}{}".format(path, i, ".pt")))

x = torch.cat(x, 0)
y = torch.cat(y, 0)

for i in range(1):
    for j in range(1):

        tgn.optimize(x, y, batch_size=200, iterations=1500, alpha=0.0001)
        torch.cuda.empty_cache()

path = "models/Model-R{}-C{}/".format(5, 128)
filename = "Model-R{}-C{}-V{}.pt".format(5, 128, "trained")
torch.save(tgn.state_dict(), path+filename)
