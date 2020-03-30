import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt

"""
for res in [3, 5, 8, 12]:
    for chan in [16, 32, 64, 128, 256]:
        pass
"""

torch.cuda.empty_cache()

value_net = ValueNetwork(alpha=0.0001, num_res=5, num_channel=128)
print(value_net.device)


x = []
y = []

x_path = "data/ogs_tensor_games/DataX"
y_path = "data/ogs_tensor_games/DataY"
for i in range(75):
    try:
        x.append(torch.load(x_path+str(i)+".pt"))
        y.append(torch.load(y_path+str(i)+".pt"))
    except:pass


x = torch.cat(x).float()
rand_perm = torch.randperm(x.shape[0])
x = x[rand_perm]
y = torch.cat(y).float()
y = y[:,82].reshape(y.shape[0], -1)[rand_perm]

print(x.shape[0])

ts = int(x.shape[0]*0.8)

x_t, y_t = x[ts:], y[ts:]
x, y = x[0:ts], y[0:ts]

value_net.optimize(x, y, x_t, y_t, batch_size=64,
 iterations=3, alpha=0.01, test_interval=5)

plt.plot(value_net.historical_loss)
plt.plot(value_net.test_accuracies)
plt.plot(value_net.test_losses)
plt.show()
