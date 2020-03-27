import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt


torch.cuda.empty_cache()

value_net = ValueNetwork(alpha=0.001, num_res=8, num_channel=32)
print(value_net.device)


x = []
y = []

x_path = "data/aya_tensor_dataset/DataX"
y_path = "data/aya_tensor_dataset/DataY"
for i in range(100):

    x.append(torch.load(x_path+str(i)+".pt")[-20:])
    y.append(torch.load(y_path+str(i)+".pt")[-20:])



x = torch.cat(x).float()
rand_perm = torch.randperm(x.shape[0])
x = x[rand_perm]
y = torch.cat(y).float()
y = y[:,82].reshape(y.shape[0], -1)[rand_perm]


value_net.optimize(x, y, batch_size=64, iterations=3, alpha=0.01)


plt.plot(value_net.historical_loss)
plt.show()

torch.save(value_net.state_dict(), "models/value-net/VN-R3-C128-V1.pt")


# process dataset
