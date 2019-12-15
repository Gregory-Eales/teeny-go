import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt


torch.cuda.empty_cache()

value_net = ValueNetwork(alpha=0.005, num_res=3, num_channel=128)
#value_net.load_state_dict(torch.load("models/value-net/VN-R5-C64-V1.pt"))
print(value_net.device)


x = []
y = []

x_path = "data/aya_dataset/DataX"
y_path = "data/aya_dataset/DataY"
for i in range(2000):

    x.append(torch.load(x_path+str(i)+".pt")[-20:])
    y.append(torch.load(y_path+str(i)+".pt")[-20:])



x = torch.cat(x).float()
rand_perm = torch.randperm(x.shape[0])
x = x[rand_perm]
y = torch.cat(y).float()
y = y[:,82].reshape(y.shape[0], -1)[rand_perm]


value_net.optimize(x, y, batch_size=16, iterations=50, alpha=0.01)


plt.plot(value_net.historical_loss)
plt.show()

torch.save(value_net.state_dict(), "models/value-net/VN-R3-C128-V1.pt")
