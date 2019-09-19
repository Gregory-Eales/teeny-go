from utils.trainer import Trainer
from teeny_go.teeny_go_network import TeenyGoNetwork

import time

import torch

torch.cuda.empty_cache()


tgn = TeenyGoNetwork(num_channels=64, num_res_blocks=5, is_cuda=True)
tgn.cuda()
trainer = Trainer(network=tgn)

t = time.time()
trainer.train_self_play(num_games=2500, is_cuda=True, iterations=150)

print("Played Through:", 100, "games in", round(time.time()-t, 3), "seconds")
