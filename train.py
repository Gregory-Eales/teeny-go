import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import argparse
from model.model import Model

import pandas as pd

def load_dataset(path, has_rand=False, num_games=992):

    x = []
    y = []
    reward = []

    x_path = path + "state-"
    y_path = path + "action-"
    reward_path = path + "reward-"

    print('loading games:')
    for i in tqdm(range(num_games)):
        try:
            x.append(torch.load(x_path+str(i)+".pt"))
            y.append(torch.load(y_path+str(i)+".pt"))
            reward.append(torch.load(reward_path+str(i)+".pt"))
            
        except:
            pass

    x = torch.cat(x).to(dtype=torch.int8)
    y = torch.cat(y).to(dtype=torch.int8)
    # shape is n, needs to be, n, 1
    reward = torch.cat(reward).to(dtype=torch.int8).unsqueeze(1)

    print(x.shape, y.shape, reward.shape)
    
    for i in range(82):
        print(f'% moves {i}', round(100*torch.sum(y[:, i]).item()/y.shape[0], 2))
    

    return x, y, reward


def main():
    
    print('loading dataset...')
    x, y, reward = load_dataset("data/computer-go-tensors/", has_rand=True, num_games=500000)


    # use the last 10% of the data for validation
    x_val = x[-4000:].float()
    y_val = y[-4000:].float()
    reward_val = reward[-4000:].float()


     # only use 10% of the data
    x = x[:-4100]
    y = y[:-4100]
    reward = reward[:-4100]


    rand_perm = torch.randperm(x.shape[0])
    x = x[rand_perm]
    y = y[rand_perm]
    reward = reward[rand_perm]

    print(x.shape, y.shape)

    model = Model(res_blocks=12, num_channels=256, in_channels=4)
    model = model.float()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"num params: {num_params}")

    print('got training and validation split')
    print(f"cuda is available: {torch.cuda.is_available()}")


    x_val = x_val.cuda()
    y_val = y_val.cuda()
    reward_val = reward_val.cuda()

    model = model.cuda()

    print(f"loaded {x.shape[0]} training samples and {x_val.shape[0]} validation samples")

    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()

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


    import torch.nn.functional as F

    def policy_gradient_loss(logits, one_hot_targets):
        log_probs = torch.log_softmax(logits, dim=1) * one_hot_targets        
        loss = -torch.sum(log_probs, dim=1)
        loss = torch.mean(loss)
        return loss

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
        reward_batch = reward[rand_perm[:batch_size*1000]].float().cuda()

        for i in tqdm(range(len(x_batch)//batch_size), desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            i1, i2 = i*batch_size, (i+1)*batch_size
            logits, _ = model.forward(x_batch[i1:i2])
            policy_loss = policy_gradient_loss(logits, y_batch[i1:i2])
            loss = policy_loss# + 0.01 * value_loss_fn(value, reward_batch[i1:i2])
            loss.backward()
            optimizer.step()

            batch_loss += loss.item() / 1000

        scheduler.step()

        losses.append(batch_loss)


        # make a prediction with no gradient
        with torch.no_grad():
             # calculate the accuracy
            logits, _ = model.forward(x_val)

            # use softmax to get the probabilities
            y_acc = torch.softmax(logits, dim=1)

            test_loss = policy_gradient_loss(logits, y_val)# + 0.01 * value_loss_fn(val, reward_val)
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
            # val_acc = (val == reward_val).sum().item()/reward_val.shape[0]
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


        #save model as simple-model.pt
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
