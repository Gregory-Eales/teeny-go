from utils.trainer import Trainer
from teeny_go.teeny_go_network import TeenyGoNetwork

import time


tgn = TeenyGoNetwork(num_channels=32, num_res_blocks=3, is_cuda=False)
#tgn.cuda()
trainer = Trainer(network=tgn)

t = time.time()
trainer.train_self_play(num_games=100, is_cuda=False, iterations=1)

print("Played Through:", 1000, "games in", round(time.time()-t, 3), "seconds")
