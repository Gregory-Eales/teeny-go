from utils.trainer import GoTrainer
from teeny_go.teeny_go_network import TeenyGoNetwork

import time


tgn = TeenyGoNetwork(num_channels=32, num_res_blocks=3, cuda=True)

gt = GoTrainer(network=tgn)

t = time.time()
gt.train_self_play(num_games=20, cuda=True)

print("Played Through:", 20, "games in", round(time.time()-t, 3), "seconds")
