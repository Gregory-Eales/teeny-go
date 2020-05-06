from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os



path = "logs/value_net/VN-2K-Samples/"


files = os.listdir(path)

file_name = "VN-R{}-C{}-{}.csv"

legend_name = "R{}-C{}"

file_types = ["Test-Accuracy", "Test-Loss", "Train-Loss"]

data_name = ["accuracy", "loss", "loss"]

data_unit = ["(%)", "(MSE)", "(MSE)"]

plot_name = ["Validation Accuracy", "Validation Loss", "Training Loss"]

legend_loc = [2, 2, 1]


figure, axes = plt.subplots(nrows=3, ncols=3)



for i, ft in enumerate(file_types):
	for j, R in enumerate([5, 8, 12]):
		for C in [64, 128, 256]:
			
			df = pd.read_csv(path+file_name.format(R, C, file_types[i]))
			ln = legend_name.format(R, C)
			axes[i, j].plot("iteration", data_name[i], data=df, label=ln)


		axes[i, j].legend(loc=legend_loc[i], fontsize=4)
		axes[i, j].set_title("R"+str(R)+" "+plot_name[i], fontsize=6)
		axes[i, j].tick_params(axis='x', labelsize=6)
		axes[i, j].tick_params(axis='y', labelsize=6)
		axes[i, j].set_xlabel("Batch", fontsize=6)
		axes[i, j].set_ylabel(data_name[i]+data_unit[i], fontsize=6)

#figure.tight_layout()

figure.suptitle("Value Network Grid Search", fontsize=10)
figure.tight_layout(w_pad=1.0, h_pad=0.5, rect=(-0.01, 0, 1, 0.95))


plt.show()


"""
fig = plt.figure()
# create a 4 plots and use tuple unpacking to name everyplot
(ax1, ax2), (ax3, ax4) = fig.subplots(2,2)
ax1.plot([1,2,3], color = "red")
ax2.plot([3,2,1], color = "blue")
ax3.plot([4,4,4], color = "orange")
ax4.plot([5,4,5], color = "black")
plt.tight_layout()
"""