from teeny_go_net import TeenyGoNetwork
import os
from tqdm import tqdm
import numpy as np
#from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch

tgn = TeenyGoNetwork()

cost = []
for i in range(1):
    X_train = []
    y_train = []
    for j in [1, 2]:
        X_train.append(np.load("X_train" + str(j) + ".npy"))
        y_train.append(np.load("y_train" + str(j) + ".npy"))
    X_train = np.reshape(np.concatenate(X_train), [61383, 1, 9, 9])
    y_train = np.concatenate(y_train)
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_train = torch.from_numpy(X_train).float()
    y_train =  torch.from_numpy(y_train).float()
    print(X_train.shape)
    cost = tgn.optimize(X_train[0:100], y_train[0:100], batch_size=1, iterations=10)



plt.plot(cost)
# summarize history for loss
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Model 1'], loc='upper left')
plt.show()
