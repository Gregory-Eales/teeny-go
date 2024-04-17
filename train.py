import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import argparse
from model.model import Model

import pandas as pd


from utils.training import load_dataset, joint_loss, shuffle_dataset, print_model_params, 


import torch.nn.functional as F



def main():
    
    print('loading training dataset...')
    x, y, r = load_dataset("data/computer-go-tensors/", start_idx=100, end_idx=30_000)
    x, y, r = shuffle_dataset(x, y, r)

    print('loading validation dataset...')
    x_val, y_val, r_val = load_dataset("data/computer-go-tensors/", start_idx=0, end_idx=100)
    x_val, y_val, r_val = x_val.cuda(), y_val.cuda(), r_val.cuda()

    print(f"loaded {x.shape[0]} training samples and {x_val.shape[0]} validation samples")
   
    
    model = Model(res_blocks=12, num_channels=256, in_channels=4).float().cuda()
    print_model_params(model)
    print('got training and validation split')
    print(f"cuda is available: {torch.cuda.is_available()}")

    # define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


    losses = []
    test_losses = []
    policy_accuracies = []
    value_accuracies = []
    train_accuracies = []
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')


    # with columns ['epoch', 'loss', 'test_loss', 'policy_accuracy', 'value_accuracy']
    data_hist = pd.DataFrame(columns=['epoch', 'loss', 'test_loss', 'policy_accuracy', 'value_accuracy'])

    # add x, y gridlines
    # train the model
    batch_size = 64
    for epoch in range(0, 10000):

        batch_loss = 0
        rand_perm = torch.randperm(x.shape[0])

        x_batch = x[rand_perm[:batch_size*1000]].float().cuda()
        y_batch = y[rand_perm[:batch_size*1000]].float().cuda()
        r_batch = r[rand_perm[:batch_size*1000]].float().cuda()

        for i in tqdm(range(len(x_batch)//batch_size), desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            i1, i2 = i*batch_size, (i+1)*batch_size
            logits, value = model.forward(x_batch[i1:i2])
            loss, pi_loss, v_loss = joint_loss(logits, value, y_batch[i1:i2], r_batch[i1:i2], alpha=0.01)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item() / 1000

        scheduler.step()

        losses.append(batch_loss)


        # make a prediction with no gradient
        with torch.no_grad():
             # calculate the accuracy
            logits, _ = model.forward(x_val)
            y_acc = torch.softmax(logits, dim=1)

            test_loss, _, _ = joint_loss(logits, value, y_val, r_val, alpha=0.01)
            test_losses.append(test_loss.item())

            y_acc = torch.argmax(y_acc, dim=1)
            acc = (y_acc == torch.argmax(y_val, dim=1)).sum().item()/y_val.shape[0]
            policy_accuracies.append(acc*100)

            # get the value accuracy
            # if > 0.1 then 1
            # if < -0.1 then -1
            # else 0

            # val = torch.where(val > 0.05, torch.tensor([1.]).cuda(), val)
            # val = torch.where(val < -0.05, torch.tensor([-1.]).cuda(), val)
            # val = torch.where((val > -0.05) & (val < 0.05), torch.tensor([0.]).cuda(), val)
            # val_acc = (val == r_val).sum().item()/r_val.shape[0]
            # value_accuracies.append(val_acc*100)

            print(f"Epoch: {epoch}, Loss: {loss.item()}, Policy Accuracy: {acc*100}%")#, Value Accuracy: {val_acc*100}%")


        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        # add title
        ax.set_title('Training Loss + Accuracy')    
        ax.plot(policy_accuracies, label="Policy Accuracy")
        #ax.plot(value_accuracies, label="Value Accuracy")
        max_loss = max(losses)
        ax.plot([100*l/max_loss for l in losses], label="Loss")
        max_loss = max(test_losses)
        ax.plot([100*l/max_loss for l in test_losses], label="Test Loss")
        plt.legend()

        # we are going headless so save the plot as a file
        plt.savefig("metrics/pi-training-loss-test.png")
        ax.clear()

        #if accuracy current accuracy is max test accuracy then save the model and epoch is a mult of 10
        if epoch % 50 == 0:
            torch.save(model, "models/pi-model-r12-c256-e{epoch}.pt".format(epoch=epoch))
            
        # update the data_hist and save using the lists
        data_hist = pd.DataFrame({'epoch': [e for e in range(epoch+1)], 'loss': losses, 'test_loss': test_losses, 'policy_accuracy': policy_accuracies})
        # save to file
        data_hist.to_csv("metrics/pi-training-metrics.csv")

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
