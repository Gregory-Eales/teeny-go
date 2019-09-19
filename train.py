from utils.trainer import Trainer
from teeny_go.teeny_go_network import TeenyGoNetwork

import time


tgn = TeenyGoNetwork(num_channels=32, num_res_blocks=3, is_cuda=True)
tgn.cuda()
trainer = Trainer(network=tgn)

t = time.time()
trainer.train_self_play(num_games=2000, is_cuda=True, iterations=5)

print("Played Through:", 100, "games in", round(time.time()-t, 3), "seconds")
