from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal
import os

"""
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
			
			df = pd.read_csv(path+file_name.format(R, C, ft))
			ln = legend_name.format(R, C)
			axes[i, j].plot("iteration", data_name[i], data=df, label=ln)


		axes[i, j].legend(loc=legend_loc[i], fontsize=4)
		axes[i, j].set_title(str(R)+" Res Block"+" "+plot_name[i], fontsize=6)
		axes[i, j].tick_params(axis='x', labelsize=6)
		axes[i, j].tick_params(axis='y', labelsize=6)
		axes[i, j].set_xlabel("Batch", fontsize=6)
		axes[i, j].set_ylabel(data_name[i]+data_unit[i], fontsize=6)

#figure.tight_layout()

figure.suptitle("Value Network Model Comparison", fontsize=10)
figure.tight_layout(w_pad=1.0, h_pad=0.5, rect=(-0.01, 0, 1, 0.95))

plt.show()

################################################################################
path = "logs/policy_net/PN-2K-Samples/"
files = os.listdir(path)
file_name = "PN-R{}-C{}-{}.csv"
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
			
			df = pd.read_csv(path+file_name.format(R, C, ft))
			ln = legend_name.format(R, C)

			if i == 0:
				y = scipy.signal.savgol_filter(df[data_name[i]], 91, 5)
				axes[i, j].plot(df["iteration"], y, label=ln)

			else:
				axes[i, j].plot("iteration", data_name[i], data=df, label=ln)


		axes[i, j].legend(loc=legend_loc[i], fontsize=4)
		axes[i, j].set_title(str(R)+" Res Block"+" "+plot_name[i], fontsize=6)
		axes[i, j].tick_params(axis='x', labelsize=6)
		axes[i, j].tick_params(axis='y', labelsize=6)
		axes[i, j].set_xlabel("Batch", fontsize=6)
		axes[i, j].set_ylabel(data_name[i]+data_unit[i], fontsize=6)

#figure.tight_layout()

figure.suptitle("Policy Network Model Comparison", fontsize=10)
figure.tight_layout(w_pad=1.0, h_pad=0.5, rect=(-0.01, 0, 1, 0.95))

plt.show()
"""
################################################################################

path = "logs/policy_net/PN-40K-Final/"
file_name = "PN-R12-C256-{}.csv"
plot_name = ["Validation Accuracy", "Validation Loss", "Training Loss"]
legend_name = ["validation accuracy", "validation loss", "training loss"]
data_name = ["accuracy", "loss", "loss"]
data_unit = ["(%)", "(MSE)", "(MSE)"]

for i, ft in enumerate(["Test-Accuracy", "Test-Loss", "Train-Loss"]):
	
	df = pd.read_csv(path+file_name.format(ft))

	if i == 0:
		y = scipy.signal.savgol_filter(df[data_name[i]], 51, 9)
		plt.plot(df["iteration"], y, label=legend_name[i])

	else:
		plt.plot("iteration", data_name[i], data=df, label=legend_name[i])


plt.title("Policy Network Training")
plt.xlabel("Batch")
plt.ylabel("Loss(MSE) / Accuracy(%)")
plt.legend(fontsize=8)
plt.show()


################################################################################

path = "logs/value_net/VN-40K-Final/"
file_name = "VN-R12-C256-{}.csv"
plot_name = ["Validation Accuracy", "Validation Loss", "Training Loss"]
legend_name = ["validation accuracy", "validation loss", "training loss"]
data_name = ["accuracy", "loss", "loss"]
data_unit = ["(%)", "(MSE)", "(MSE)"]

for i, ft in enumerate(["Test-Accuracy", "Test-Loss", "Train-Loss"]):
	
	df = pd.read_csv(path+file_name.format(ft))

	if i == 0:
		y = scipy.signal.savgol_filter(df[data_name[i]], 51, 2)
		plt.plot(df["iteration"], y, label=legend_name[i])

	if i == 1:
		y = scipy.signal.savgol_filter(df[data_name[i]], 101, 2)
		plt.plot(df["iteration"], y, label=legend_name[i])

	if i == 2:
		y = scipy.signal.savgol_filter(df[data_name[i]], 51, 2)
		plt.plot(df["iteration"], y, label=legend_name[i])

	#else:
		#plt.plot("iteration", data_name[i], data=df, label=legend_name[i])


plt.title("Value Network Training")
plt.xlabel("Batch")
plt.ylabel("Loss(MSE) / Accuracy(%)")
plt.legend(fontsize=8, loc=0)
plt.show()







