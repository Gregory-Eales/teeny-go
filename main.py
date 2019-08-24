from teeny_go.go_trainer import GoTrainer

import torch




tg = TeenyGo()


torch.save(tg.network,f="Model-A1")
