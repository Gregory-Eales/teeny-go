
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

from teeny_go.teeny_go import TeenyGo
from utils.trainer import Trainer

def main(hparams):


	model = TeenyGo()
	trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)