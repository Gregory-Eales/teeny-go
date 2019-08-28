#from teeny_go.go_trainer import GoTrainer
#import logging
import torch



x = torch.randn(100, 11, 9, 9)

y = torch.randn(100, 1, 9, 9)


torch.save(x, 'x.pt')
torch.save(y, 'y.pt')
