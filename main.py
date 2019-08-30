#from teeny_go.go_trainer import GoTrainer
#import logging
import torch
from teeny_go.go_trainer import GoTrainer
from teeny_go.teeny_go_network import TeenyGoNetwork


def load_data():
    pass



tgn = TeenyGoNetwork()
tgn.load_state_dict("Models\Model_R5_C64_V0.pt")
