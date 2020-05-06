import torch
from tqdm import tqdm
from decimal import Decimal

class Block(torch.nn.Module):

    def __init__(self, num_channel):
        super(Block, self).__init__()
        self.pad1 = torch.nn.ZeroPad2d(1)
        self.conv1 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_channel)
        self.relu1 = torch.nn.ReLU()
        self.pad2 = torch.nn.ZeroPad2d(1)
        self.conv2 = torch.nn.Conv2d(num_channel, num_channel, kernel_size=3)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_channel)
        self.relu2 = torch.nn.ReLU()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.batch_norm1(out)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = out + x
        out = self.relu2(out)
        return out
    
    
class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, num_res=3, num_channel=3):
        super(PolicyNetwork, self).__init__()

        #self.input_channels = num_channel
        self.state_channel = 3
        self.num_res = num_res
        self.res_block = torch.nn.ModuleDict()
        self.num_channel = num_channel
        self.historical_loss = []

        # network metrics
        self.training_losses = []
        self.test_losses = []
        self.training_accuracies = []
        self.test_accuracies = []
        self.test_iteration = []

        self.model_name = "VN-R" + str(self.num_res) + "-C" + str(self.num_channel)

        self.define_network()
        # define optimizer
        self.optimizer = torch.optim.Adam(lr=alpha, params=self.parameters())
        self.loss = torch.nn.BCELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def define_network(self):
        #policy head
        self.policy_conv = torch.nn.Conv2d(self.num_channel, 2, kernel_size=1)
        self.policy_batch_norm = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
        self.policy_fc = torch.nn.Linear(2*9*9, 82)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

        # network start
        self.pad = torch.nn.ZeroPad2d(1)
        self.conv = torch.nn.Conv2d(self.state_channel, self.num_channel, kernel_size=3)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
        self.relu = torch.nn.ReLU()

        for i in range(1, self.num_res+1):
            self.res_block["r"+str(i)] = Block(self.num_channel)

    def forward(self, x):
        out = torch.Tensor(x.float()).to(self.device)

        out = self.pad(out)
        out = self.conv(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        for i in range(1, self.num_res+1):
            out = self.res_block["r"+str(i)](out)

        # policy head
        out = self.policy_conv(out)
        out = self.policy_batch_norm(out)
        out = self.relu(out)
        out = out.reshape(-1, 2*9*9)
        out = self.policy_fc(out)
        out = self.sigmoid(out)
        return out.to(torch.device('cpu:0'))
    
    def optimize(self, x, y, x_t, y_t,
     batch_size=16, iterations=10, alpha=0.1, test_interval=1000, save=False):
        
        x_t = x_t.float()
        y_t = y_t.float()
        
        a = "-A" + ('%E' % Decimal(str(alpha)))[-1]

        model_name = "PN-R" + str(self.num_res) + "-C" + str(self.num_channel)
        self.model_name = model_name
        model_path = "models/policy_net/{}".format(model_name)
        log_path = "logs/policy_net/{}/".format(model_name)

        num_batch = x.shape[0]//batch_size
        remainder = x.shape[0]%batch_size
            
        print("Training Model:", model_name)
        
        # Train netowork
        for iter in tqdm(range(iterations)):
            # save the model after each iteration
            if save:
                #torch.save(self.state_dict(), self.model_name+"-P{}.pt".format(iter))
                pass

            for i in range(num_batch):
                prediction = self.forward(x[i*batch_size:(i+1)*batch_size])
                loss = self.loss(prediction, y[i*batch_size:(i+1)*batch_size].float())
                del(prediction)
                self.historical_loss.append(loss.detach())
                self.optimizer.zero_grad()
                if iter == 0 and i == 0: loss.backward(retain_graph=True)
                else: loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()

                if (iter*num_batch + i)%test_interval == 0 and save:
                    self.test_model(x_t, y_t)
                    self.test_iteration.append(iter*num_batch + i)

              
                del(loss)

            torch.cuda.empty_cache()
            if save:
                self.save_metrics()
        
        #torch.save(self.state_dict(), self.model_name+"-P{}.pt".format("Final"))
                
    def test_model(self, x_t, y_t):
        
        prediction = self.forward(x_t)
        test_accuracy = self.get_test_accuracy(prediction, y_t)
        test_loss = self.get_test_loss(prediction, y_t)

        m = len(self.historical_loss)
        l = torch.Tensor(self.historical_loss)
        training_loss = torch.sum(l)/m
        
        del(prediction)
        torch.cuda.empty_cache()
        self.historical_loss = []

        self.test_accuracies.append(test_accuracy.detach().type(torch.float16))
        self.test_losses.append(test_loss.detach().type(torch.float16))
        self.training_losses.append(training_loss.type(torch.float16))


    def get_test_accuracy(self, prediction, y_t):

        c = torch.zeros(y_t.shape[0], y_t.shape[1])
        
        c[prediction == prediction.max(dim=0)[0]] = 1
        c[prediction != prediction.max(dim=0)[0]] = 0

        correct_percent = torch.sum(c*y_t) / y_t.shape[0]

        return correct_percent

    def get_test_loss(self, prediction, y_t):
        return self.loss(prediction, y_t)

    def save_training_loss(self, path=""):
        file_name = path + self.model_name+"-Train-Loss.csv"
        file = open(file_name, "w")
        file.write("iteration,loss\n")
        for iteration, loss in enumerate(self.training_losses):
            file.write("{},{}\n".format(iteration, loss))
        file.close()
        
    def save_test_loss(self, path=""):
        assert len(self.test_losses) == len(self.test_iteration)
        file_name = path + self.model_name+"-Test-Loss.csv"
        file = open(file_name, "w")
        file.write("iteration,loss\n")
        for i, loss in enumerate(self.test_losses):
            file.write("{},{}\n".format(self.test_iteration[i], loss))
        file.close()

    def save_test_accuracy(self, path=""):
        assert len(self.test_accuracies) == len(self.test_iteration)
        file_name = path + self.model_name+"-Test-Accuracy.csv"
        file = open(file_name, "w")
        file.write("iteration,accuracy\n")
        for i, acc in enumerate(self.test_accuracies):
            file.write("{},{}\n".format(self.test_iteration[i], acc))
        file.close()

    def save_metrics(self):
        self.save_training_loss(self.model_name)
        self.save_test_loss(self.model_name)
        self.save_test_accuracy(self.model_name)
        
    def save(self):
        torch.save(self.state_dict(), self.model_name+".pt")
        
def generate_data(num_games=1000):
    x = []
    y = []
    x_path = "/kaggle/input/godataset/new_ogs_tensor_games/DataX"
    y_path = "/kaggle/input/godataset/new_ogs_tensor_games/DataY"
    for i in range(num_games):
        try:
            x.append(torch.load(x_path+str(i)+".pt"))
            y.append(torch.load(y_path+str(i)+".pt"))
        except:pass
    
    x = torch.cat(x)
    y = torch.cat(y)

    ts = int(x.shape[0]*0.95)
    x_t, y_t = x[ts:], y[ts:][:,0:82].reshape(y[ts:].shape[0], 82)
    rand_perm = torch.randperm(x_t.shape[0])
    x_t, y_t = x_t[rand_perm], y_t[rand_perm]
    x_t = x_t[0:250]
    y_t = y_t[0:250]
    
    rand_perm = torch.randperm(x[0:ts].shape[0])
    x = x[0:ts][rand_perm]
    y = y[0:ts][:,0:82].reshape(y[0:ts].shape[0], 82)[rand_perm]

    print("Data Samples:", x.shape[0])
    
    return x, y, x_t, y_t

def main():
    import torch
    
    x, y, x_t, y_t = generate_data(num_games=2000)
    
    for res in [5, 8, 12, 15]:
        for chan in [64, 128, 256]:
            # clear cache and define network
            torch.cuda.empty_cache()
            policy_net = PolicyNetwork(alpha=0.000001, num_res=res, num_channel=chan)

            # train model
            policy_net.optimize(x, y, x_t, y_t, batch_size=64,
                                       iterations=20, alpha=0.000001, test_interval=10, save=True)
            #value_net.save()

            # clear cache and remove old network
            del(policy_net)
            torch.cuda.empty_cache()           

if __name__ == "__main__":
    main()