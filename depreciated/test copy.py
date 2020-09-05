import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt


torch.cuda.empty_cache()

value_net = ValueNetwork(alpha=0.01, num_res=5, num_channel=64)
value_net.load_state_dict(torch.load("models/value-net/VN-R5-C64-V1.pt"))
print(value_net.device)


x = []
y = []

x_path = "data/aya_dataset/DataX"
y_path = "data/aya_dataset/DataY"
for i in range(1000, 1100):

    x.append(torch.load(x_path+str(i)+".pt"))
    y.append(torch.load(y_path+str(i)+".pt"))


x = torch.cat(x).float()
rand_perm = torch.randperm(x.shape[0])
x = x[rand_perm]
y = torch.cat(y).float()
y = y[:,82].reshape(y.shape[0], -1)[rand_perm]

p = value_net.forward(x)
c = torch.zeros(y.shape[0], y.shape[1])
c[p<-0.33] = -1
c[p>0.33] = 1

correct_percent = torch.sum(((c+y)/2)**2) / y.shape[0]
print(correct_percent)
