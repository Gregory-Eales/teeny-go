from utils.trainer import Trainer
from teeny_go.teeny_go_network import TeenyGoNetwork

import time

import torch

torch.cuda.empty_cache()


tgn = TeenyGoNetwork(num_channels=128, num_res_blocks=3, is_cuda=True)
tgn.cuda()
trainer = Trainer(network=tgn)


trainer.train_self_play(num_games=500, is_cuda=True, iterations=250)



"""
for i in range(10):
    for j in range(10):

        file_num = 150-j

        path = "data/Model-R{}-C{}/".format(5, 64)
        filenameX = "Model-R{}-C{}-V{}-DataX.pt".format(5, 64, file_num)
        filenameY = "Model-R{}-C{}-V{}-DataY.pt".format(5, 64, file_num)


        x = torch.load(path+filenameX)
        y = torch.load(path+filenameY)

        x = x.cuda().type(torch.cuda.FloatTensor)
        y = y.cuda().type(torch.cuda.FloatTensor)

        tgn.optimize(x, y, batch_size=5000, iterations=1, alpha=0.01)

        torch.cuda.empty_cache()

        del(x)
        del(y)

        torch.cuda.empty_cache()


path = "models/Model-R{}-C{}/".format(5, 64)
filename = "Model-R{}-C{}-V{}.pt".format(5, 64, "trained")
torch.save(tgn.state_dict(), path+filename)
"""
