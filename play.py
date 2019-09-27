from utils.viewer import Viewer
from teeny_go.teeny_go_network import TeenyGoNetwork

import torch

import time


tgn = TeenyGoNetwork(num_channels=256, num_res_blocks=3, is_cuda=False)

tgn.load_state_dict(torch.load("models/Model-R3-C256/Model-R3-C256-VSL0.pt"))

viewer = Viewer()

viewer.human_vs_ai(tgn)
