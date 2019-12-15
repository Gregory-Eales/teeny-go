import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt


torch.cuda.empty_cache()

value_net = ValueNetwork(alpha=0.01, num_res=3, num_channel=32)
print(value_net.device)


x = []
y = []

x_path = "data/aya_dataset/DataX"
y_path = "data/aya_dataset/DataY"
for i in range(100):

    x.append(torch.load(x_path+str(i)+".pt"))
    y.append(torch.load(y_path+str(i)+".pt"))



x = torch.cat(x).float()
rand_perm = torch.randperm(x.shape[0])
x = x[rand_perm]
y = torch.cat(y).float()
y = y[:,82].reshape(y.shape[0], -1)[rand_perm]


value_net.optimize(x, y, batch_size=16, iterations=5, alpha=0.01)

print(value_net.forward(x).shape)
print(y.shape)


plt.plot(value_net.historical_loss)
plt.show()

print(y[-10:])
print(value_net.forward(x[-10:]))


torch.save(value_net.state_dict(), "models/Model-V1-R3-C32.pt")
