import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser, Namespace

#from teeny_go.teeny_go import TeenyGo
from teeny_go.joint_network import JointNetwork
#from utils.go_dataset import GoDataset


import torch

print(torch.cuda.is_available())


def main(hparams):

   model = JointNetwork(hparams)
   trainer = Trainer(gpus=args.gpu, max_epochs=args.max_epochs, auto_lr_find=True)
   trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    
    parser = ArgumentParser()

    # training params
    parser.add_argument("--gpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--early_stopping", type=bool, default=True, help="whether early stopping is used or not")
    parser.add_argument("--max_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of training batch")
    parser.add_argument("--lr", type=int, default=0.001, help="learning rate")
    #parser.add_argument("--accumulate_grad_batches", type=int, default=64, help="grad batches")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="")
    parser.add_argument("--auto_lr_find", type=str, default="lr", help="finds the optimal lr rate")

    # general network params
    parser.add_argument("--in_channels", type=int, default=6, help="number of input channels")
    parser.add_argument("--kernal_size", type=int, default=3, help="convolutional kernal size")
    parser.add_argument("--num_channels", type=int, default=256, help="number of channels in the res blocks")
    parser.add_argument("--num_res_blocks", type=int, default=16, help="number of residual blocks")

    # value network params
    parser.add_argument("--value_accuracy_boundry", type=float, default=0.1,
     help="threshold before value prediction is considered correct")
   
    # dataset params
    parser.add_argument("--num_games", type=int, default=23000)
    parser.add_argument("--data_split", type=list, default=[0.9, 0.05, 0.05], help="train, validation, test split")
    parser.add_argument("--data_path", type=str, default="data/ogs_dan_tensor_games/", help="path to dataset")
                     
    args = parser.parse_args()

    main(args)