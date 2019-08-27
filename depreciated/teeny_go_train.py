from teeny_go_net import TeenyGoNetwork
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch

tgn = TeenyGoNetwork()

tgn = tgn.cuda()

cost = []
for i in range(1):
    X_train = []
    y_train = []
    for j in [1, 2]:
        X_train.append(np.load("X_train" + str(j) + ".npy"))
        #y_train.append(np.load("y_train" + str(j) + ".npy"))
    X_train = np.reshape(np.concatenate(X_train), [61383, 1, 9, 9])
    #y_train = np.concatenate(y_train)
    #X_train, y_train = shuffle(X_train[0:2500], y_train[0:2500], random_state=0)


    x1 = X_train
    y1 = np.ones([X_train.shape[0], 1])
    x2 = X_train * -1
    y2 = np.ones([X_train.shape[0], 1]) * -1
    print(X_train.shape[0])
    x = np.concatenate([x2, x1])
    y = np.concatenate([y2, y1])
    cost = []
    for j in range(5):
        for i in range(4):
            X_train, y_train = shuffle(x[i*2500:(i+1)*2500], y[i*2500:(i+1)*2500], random_state=0)
            #print(y_train[0:100])
            break
            X_train = torch.from_numpy(X_train).float().cuda()
            y_train =  torch.from_numpy(y_train).float().cuda()
            #print(X_train.shape)

            print(X_train[0])
            print(y_train[0])

            costy = tgn.optimize(X_train, y_train, batch_size=1000, iterations=1)
            cost += costy


plt.plot(cost)
# summarize history for loss
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Model 1'], loc='upper left')
plt.show()
