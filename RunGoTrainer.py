from OptimizedGoTrainer import GoTrainer
from SGF_Loader import get_data
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

GT = GoTrainer()

#path = "GoSampleData/619920.sgf"

# get data file paths
for path in os.walk("GoSampleData"):
    paths = (path[2][0:10])
# print action
# play out game and gather data
print("Playing Out Games: ")
for i in tqdm(range(len(paths))):
    data, winner = get_data("GoSampleData/"+paths[i])
    GT.play(data=data, winner=winner)

print("Training Teeny Go Neural Net: ")
for i in tqdm(range(1000)):
    for j in range(len(GT.x_data)):
        GT.NN.x = GT.x_data[j]
        GT.NN.y = GT.y_data[j]
        GT.NN.optimize()

print(GT.x_data[5])
print(GT.y_data[5])
print(GT.NN.a["a"+str(3)])


# print cost over time.
x = []
for i in range(len(GT.NN.historical_cost)):
    x.append(i)

plt.plot(x, GT.NN.historical_cost)
plt.show()
