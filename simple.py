import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_dataset(path, num_games=6000):

    x = []
    y = []
    x_path = path + "DataX"
    y_path = path + "DataY"

    for i in range(num_games):
        try:
            x.append(torch.load(x_path+str(i)+".pt"))
            y.append(torch.load(y_path+str(i)+".pt"))
        except:
            pass

    x = torch.cat(x).float()
    rand_perm = torch.randperm(x.shape[0])
    x = x[rand_perm]
    y = torch.cat(y).float()
    y = y[:,:82].reshape(y.shape[0], -1)[rand_perm]

    return x, y


def main():
    
    x, y = load_dataset("data/big_15k_tensor/")
    x = x.unsqueeze(1)  # Add channel dimension to your data
    
    print(x.shape, y.shape)

    # make a small conv net
    conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
    conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
    conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
    
    fc1 = torch.nn.Linear(32*9*9, 128)
    fc2 = torch.nn.Linear(128, 82)

    model = torch.nn.Sequential(
        conv1,
        torch.nn.GELU(),
        conv2,
        torch.nn.GELU(),
        conv3,
        torch.nn.GELU(),
        torch.nn.Flatten(),
        fc1,
        torch.nn.GELU(),
        fc2,
        torch.nn.Softmax(dim=1)
    )

    # make sure model and data are float32
    model = model.float()
    x = x.float()
    y = y.float()

    # use the last 10% of the data for validation
    x_val = x[int(x.shape[0]*0.90):]
    y_val = y[int(y.shape[0]*0.90):]

    # only use 10% of the data
    x = x[:int(x.shape[0]*0.80)]
    y = y[:int(y.shape[0]*0.80)]


    rand_perm = torch.randperm(x.shape[0])
    x = x[rand_perm]
    y = y[rand_perm]

    # define the loss function
    loss_fn = torch.nn.MSELoss(reduction='mean')

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    test_accuracies = []
    train_accuracies = []
    # make a realtime plot  
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # add title
    ax.set_title('Training Loss')
    # add x, y gridlines
    # train the model
    for epoch in range(1000):
        # randomly shuffle the data at the end of each epoch
        rand_perm = torch.randperm(x.shape[0])
        x = x[rand_perm]
        y = y[rand_perm]

        for i in tqdm(range(100), desc=f"Epoch: {epoch}"):
            i1, i2 = i*128, (i+1)*128
            x_batch, y_batch = x[i1:i2], y[i1:i2]
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            #print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())


        # calculate the accuracy
        y_acc = model(x_val)
        y_acc = torch.argmax(y_acc, dim=1)
        acc = (y_acc == torch.argmax(y_val, dim=1)).sum().item()/y_val.shape[0]
        test_accuracies.append(acc*100)

        y_acc = model(x)
        y_acc = torch.argmax(y_acc, dim=1)
        acc = (y_acc == torch.argmax(y, dim=1)).sum().item()/y.shape[0]
        train_accuracies.append(acc*100)
            
        ax.plot(test_accuracies)
        ax.plot(train_accuracies)

        fig.canvas.draw()
        plt.pause(0.001)
            

    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
