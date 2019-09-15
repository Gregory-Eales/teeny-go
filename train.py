from utils.trainer import GoTrainer
from teeny_go.teeny_go_network import TeenyGoNetwork

import time


tgn = TeenyGoNetwork(num_channels=32, num_res_blocks=3, is_cuda=True)
tgn.cuda()
gt = GoTrainer(network=tgn)

t = time.time()
gt.train_self_play(num_games=5000, is_cuda=True, iterations=100)

print("Played Through:", 1000, "games in", round(time.time()-t, 3), "seconds")
