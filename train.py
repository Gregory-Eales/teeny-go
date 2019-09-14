from utils.trainer import GoTrainer
from teeny_go.teeny_go_network import TeenyGoNetwork


tgn = TeenyGoNetwork(num_channels=32, num_res_blocks=5)
tgn.double()

gt = GoTrainer(network=tgn)

gt.play_through_games(num_games=10)
