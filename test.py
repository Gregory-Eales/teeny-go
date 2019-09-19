from utils.tester import Tester
from teeny_go.teeny_go_network import TeenyGoNetwork

import torch

import time


tgn1 = TeenyGoNetwork(num_channels=32, num_res_blocks=3, is_cuda=False)
tgn2 = TeenyGoNetwork(num_channels=32, num_res_blocks=3, is_cuda=False)

tgn1.load_state_dict(torch.load("models/Model-R3-C32/Model-R3-C32-V21.pt"))
tgn1.load_state_dict(torch.load("models/Model-R3-C32/Model-R3-C32-V5.pt"))


tester = Tester()


tester.play_through_games(tgn1, tgn2, num_games=1000)
