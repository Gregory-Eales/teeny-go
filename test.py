from utils.tester import Tester
from utils.multi_tester import MultiTester
from teeny_go.teeny_go_network import TeenyGoNetwork

import torch

import time


tgn1 = TeenyGoNetwork(num_channels=32, num_res_blocks=3, is_cuda=False)
tgn2 = TeenyGoNetwork(num_channels=64, num_res_blocks=5, is_cuda=False)
tgn3 = TeenyGoNetwork(num_channels=64, num_res_blocks=5, is_cuda=False)

tgn1.load_state_dict(torch.load("models/Model-R3-C32/Model-R3-C32-V3.pt"))
tgn2.load_state_dict(torch.load("models/Model-R5-C64/Model-R5-C64-VSL1.pt"))
tgn3.load_state_dict(torch.load("models/Model-R5-C64/Model-R5-C64-VSL.pt"))


tester = Tester()

#tester.play_through_games(tgn1, tgn2, num_games=100)
tester.play_through_games(tgn2, tgn3, num_games=100)
