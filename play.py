from utils.viewer import Viewer
from teeny_go.teeny_go_network import TeenyGoNetwork

import torch

import time


tgn = TeenyGoNetwork(num_channels=64, num_res_blocks=5, is_cuda=False)

tgn.load_state_dict(torch.load("models/Model-R5-C64/Model-R5-C64-V8.pt"))

viewer = Viewer()

viewer.human_vs_ai(tgn)
