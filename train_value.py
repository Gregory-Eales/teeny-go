import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt

"""
for res in [3, 5, 8, 12]:
    for chan in [16, 32, 64, 128, 256]:
        pass
"""

torch.cuda.empty_cache()

value_net = ValueNetwork(alpha=0.0001, num_res=3, num_channel=64)
print(value_net.device)


x = []
y = []

x_path = "data/ogs_tensor_games/DataX"
y_path = "data/ogs_tensor_games/DataY"
for i in range(20):
    try:
        x.append(torch.load(x_path+str(i)+".pt"))
        y.append(torch.load(y_path+str(i)+".pt"))
    except:pass

x = torch.cat(x).float()
y = torch.cat(y).float()

ts = int(x.shape[0]*0.95)
x_t, y_t = x[ts:], y[ts:][:,82].reshape(y[ts:].shape[0], -1)

rand_perm = torch.randperm(x[0:ts].shape[0])
x = x[0:ts][rand_perm]
y = y[0:ts][:,82].reshape(y[0:ts].shape[0], -1)[rand_perm]

print("Data Samples:", x.shape[0])

value_net.optimize(x, y, x_t, y_t, batch_size=64,
 iterations=10, alpha=0.01, test_interval=10, save=True)

plt.plot(value_net.test_iteration, value_net.training_losses)
plt.plot(value_net.test_iteration, value_net.test_accuracies)
plt.plot(value_net.test_iteration, value_net.test_losses)
plt.show()
