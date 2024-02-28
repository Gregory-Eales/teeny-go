import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import argparse

def load_dataset(path, num_games=30000):

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
    
    x, y = load_dataset("data/gnu_go_tensors/", num_games=10000)
    x = x.unsqueeze(1)  # Add channel dimension to your data
    
    print(x.shape, y.shape)

    # make a small conv net
    conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
    conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
    conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
    
    fc1 = torch.nn.Linear(32*9*9, 32*9*9)
    fc2 = torch.nn.Linear(32*9*9, 128)
    fc3 = torch.nn.Linear(128, 82)

    bn1 = torch.nn.BatchNorm2d(32)
    bn2 = torch.nn.BatchNorm2d(32)

    model = torch.nn.Sequential(
        conv1,
        torch.nn.LeakyReLU(),
        conv2,
        bn1,
        torch.nn.LeakyReLU(),
        conv3,
        bn2,
        torch.nn.LeakyReLU(),
        torch.nn.Flatten(),
        fc1,
        torch.nn.LeakyReLU(),
        fc2,
        torch.nn.LeakyReLU(),
        fc3,
        torch.nn.Softmax(dim=1)
    )

    # make sure model and data are float32
    model = model.float()

    # load simple-model.pt if it exists
    try:
        model = torch.load("simple-model.pt")
        print("loaded model")
    except:
        print("failed to load model")

    x = x.float()
    y = y.float()

    # make y (n, 1) to (n)


    # rand_perm = torch.randperm(x.shape[0])
    # x = x[rand_perm]
    # y = y[rand_perm]

    # use the last 10% of the data for validation
    x_val = x[int(x.shape[0]*0.99):]
    y_val = y[int(y.shape[0]*0.99):]

    # only use 10% of the data
    x = x[:int(x.shape[0]*0.95)]
    y = y[:int(y.shape[0]*0.95)]

    print(f"loaded {x.shape[0]} training samples and {x_val.shape[0]} validation samples")


    rand_perm = torch.randperm(x.shape[0])
    x = x[rand_perm]
    y = y[rand_perm]

    # define the loss function for multi-class classification cross entropy
    loss_fn = torch.nn.CrossEntropyLoss()

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1*1e-4)

    losses = []
    test_accuracies = []
    train_accuracies = []
    # make a realtime plot  
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # add title
    ax.set_title('Training Loss')
    # add x, y gridlines
    # train the model
    batch_size = 128
    for epoch in range(1000):
        # randomly shuffle the data at the end of each epoch
        rand_perm = torch.randperm(x.shape[0])
        x = x[rand_perm]
        y = y[rand_perm]

        batch_loss = 0

        for i in tqdm(range(1000), desc=f"Epoch: {epoch}"):#range(len(x)//batch_size), desc=f"Epoch: {epoch}"):
            i1, i2 = i*batch_size, (i+1)*batch_size
            x_batch, y_batch = x[i1:i2], y[i1:i2]
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            #print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #losses.append(loss.item())

            batch_loss += loss.item()

        batch_loss /= len(x)//batch_size
        losses.append(batch_loss)


       

        # make a prediction with no gradient
        with torch.no_grad():
             # calculate the accuracy
            y_acc = model(x_val)
            y_acc = torch.argmax(y_acc, dim=1)
            acc = (y_acc == torch.argmax(y_val, dim=1)).sum().item()/y_val.shape[0]
            test_accuracies.append(acc*100)

            # y_acc = model(x)
            # y_acc = torch.argmax(y_acc, dim=1)
            # acc = (y_acc == torch.argmax(y, dim=1)).sum().item()/y.shape[0]
            # train_accuracies.append(acc*100)
            
        ax.plot(test_accuracies)
        ax.plot(train_accuracies)
        max_loss = max(losses)
        ax.plot([100*l/max_loss for l in losses])

        fig.canvas.draw()
        plt.pause(0.001)
        ax.clear()


        # save model as simple-model.pt
        torch.save(model, "simple-model.pt")
            

    plt.plot(losses)
    plt.show()

def init_go_env():
    parser = argparse.ArgumentParser(description='Go Simulation')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()
    return gym.make('gym_go:go-v0',size=args.boardsize)#, reward_method='real')


def play_model():

    # load the model if it exists
    model = None
    try:
        model = torch.load("simple-model.pt")
        print("loaded model")
        # print the number of paremeters in the model
        print(f"model has {sum([p.numel() for p in model.parameters()])} parameters")
    except:
        print("failed to load model")
        return
    
    env = gym.make("gym_go:go-v0", size=9, komi=0, reward_method="heuristic")

    state = env.reset()
    done = False
    while not done:

        # get the action from the model
        with torch.no_grad():
            state = torch.tensor(state[0] - state[1]).unsqueeze(0).unsqueeze(0).float()
            print(state.shape)
            action = model(state)
            # multiply env.valid_moves() by the action to get the valid actions
            action = action*torch.tensor(env.valid_moves()).float()
            # don't let the model pass
            #action[0][81] = 0
            action = torch.argmax(action, dim=1).item()

            
            state, reward, done, info = env.step(action)
            env.render(mode="terminal")

            # get human input
            action = input("enter action: ")
            action = action.split(" ")
            action = (int(action[0]), int(action[1]))
            state, reward, done, info = env.step(action)
            env.render(mode="terminal")

def plot_game_length():

    path = "data/gnu_go_tensors/"

    x_path = path + "DataX"
    y_path = path + "DataY"

    lengths = []

    for i in range(10000):
        try:
            x = torch.load(x_path+str(i)+".pt")

            lengths.append(x.shape[0])

        except:
            pass

    
    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Game')
    ax.set_ylabel('Length')
    # add title
    ax.set_title('Game Lengths')
    # add x, y gridlines
    #ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
    # make histogram
    ax.hist(lengths, bins=20)
    plt.show()





if __name__ == "__main__":
    main()
    #play_model()
    #plot_game_length()
