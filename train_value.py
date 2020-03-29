import torch
from teeny_go.value_network import ValueNetwork
from matplotlib import pyplot as plt


for res in [3, 5, 8, 12]:
    for chan in [16, 32, 64, 128, 256]:

        torch.cuda.empty_cache()

        value_net = ValueNetwork(alpha=0.001, num_res=res, num_channel=chan)
        print(value_net.device)


        x = []
        y = []

        x_path = "data/pro_game_dataset/DataX"
        y_path = "data/pro_game_dataset/DataY"
        for i in range(10):
            try:
                x.append(torch.load(x_path+str(i)+".pt"))
                y.append(torch.load(y_path+str(i)+".pt"))
            except:pass


        x = torch.cat(x).float()
        rand_perm = torch.randperm(x.shape[0])
        x = x[rand_perm]
        y = torch.cat(y).float()
        y = y[:,82].reshape(y.shape[0], -1)[rand_perm]

        print(y.shape)
        print(x.shape)

        value_net.optimize(x, y, batch_size=64, iterations=3, alpha=0.01)


        plt.plot(value_net.historical_loss)
        plt.show()

        torch.save(value_net.state_dict(), "models/value-net/VN-R3-C128-V1.pt")


        # process dataset
