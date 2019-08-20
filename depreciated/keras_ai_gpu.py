from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from SGF_Loader import get_data
from OptimizedGoTrainer import GoTrainer
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from keras.backend import manual_variable_initialization
from sklearn.utils import shuffle
import game_processor
# enables manual variable initialization
# manual_variable_initialization(True)

GT = GoTrainer()

# create model 1
model = Sequential()
# add model layers
model.add(Conv2D(81, kernel_size=2, activation='relu', padding="same", input_shape=(9, 9, 1)))
model.add(Conv2D(81, kernel_size=3, activation='relu', padding="same"))
model.add(Conv2D(81, kernel_size=4, activation='relu', padding="same"))
model.add(Conv2D(81, kernel_size=5, activation='relu', padding="same"))
model.add(Conv2D(81, kernel_size=6, activation='relu', padding="same"))

# create model
model = Sequential()
# add model layers
model.add(Conv2D(60, kernel_size=2, activation='tanh', padding="same", input_shape=(9, 9, 1)))
model.add(Conv2D(50, kernel_size=2, activation='tanh', padding="same"))
model.add(Conv2D(30, kernel_size=2, activation='tanh', padding="same"))

# model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(160, activation='relu', use_bias=True))
model.add(Dense(160, activation='relu', use_bias=True))
model.add(Dense(81, activation='sigmoid', use_bias=True))

# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# create model 2
model2 = Sequential()
# add model layers
model2.add(Conv2D(81, kernel_size=2, activation='relu', padding="same", input_shape=(9, 9, 1)))
model2.add(Conv2D(81, kernel_size=2, activation='relu', padding="same"))
model2.add(Conv2D(81, kernel_size=2, activation='relu', padding="same"))

# model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
model2.add(Flatten())
model2.add(Dense(160, activation='relu', use_bias=True))
model2.add(Dense(160, activation='relu', use_bias=True))
model2.add(Dense(81, activation='sigmoid', use_bias=True))

# compile model using accuracy to measure model performance
model2.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# create model 3
model3 = Sequential()
# add model layers
model3.add(Conv2D(200, kernel_size=5, activation='relu', padding="same", input_shape=(9, 9, 1)))
model3.add(Conv2D(150, kernel_size=4, activation='relu', padding="same"))
model3.add(Conv2D(120, kernel_size=3, activation='relu', padding="same"))
model3.add(Conv2D(100, kernel_size=2, activation='relu', padding="same"))
model3.add(Conv2D(80, kernel_size=1, activation='relu', padding="same"))

# model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
model3.add(Flatten())
model3.add(Dense(160, activation='relu', use_bias=True))
model3.add(Dense(81, activation='sigmoid', use_bias=True))

# compile model using accuracy to measure model performance
model3.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# create model 2
model4 = Sequential()
# add model layers
model4.add(Conv2D(16, kernel_size=4, activation='relu', padding="same", input_shape=(9, 9, 1)))
model4.add(Conv2D(16, kernel_size=4, activation='relu', padding="same"))
model4.add(Conv2D(32, kernel_size=4, activation='relu', padding="same"))
model4.add(Conv2D(32, kernel_size=3, activation='relu', padding="same"))
model4.add(Conv2D(64, kernel_size=3, activation='relu', padding="same"))
model4.add(Conv2D(64, kernel_size=2, activation='relu', padding="same"))
model4.add(Conv2D(128, kernel_size=2, activation='relu', padding="same"))
model4.add(Conv2D(128, kernel_size=1, activation='relu', padding="same"))

# model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
model4.add(Flatten())
model4.add(Dense(160, activation='relu', use_bias=True))
model4.add(Dense(160, activation='relu', use_bias=True))
model4.add(Dense(81, activation='sigmoid', use_bias=True))

# compile model using accuracy to measure model performance
model4.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])







"""
print("Training Teeny Go Neural Net: ")
X_train = []
y_train = []

for x in range(1, 5):

    X_train = []
    y_train = []
    # get data file paths
    for path in os.walk("/Users/Greg/Desktop/GoData/9x9GoData"):
        paths = (path[2][x*1000:(x+1)*1000])
    for i, path in enumerate(paths):
    	paths[i] = "/Users/Greg/Desktop/GoData/9x9GoData/" + path

    print(len(paths))

    X_train, y_train = game_processor.process_multi_sgf(paths)
        
        #X_train.append(GT.x_data[j].reshape(9, 9, 1))
        #y_train.append(GT.y_data[j].reshape(81))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    np.save("X_train"+str(x+1), X_train.reshape(X_train.shape[0], 9, 9, 1))
    np.save("y_train"+str(x+1), y_train.reshape(y_train.shape[0], 81))
"""

cost = []
for i in range(10):

	X_train = []
	y_train = []
	for j in [1, 2]:
		X_train.append(np.load("X_train" + str(j) + ".npy"))
		y_train.append(np.load("y_train" + str(j) + ".npy"))

	

	# model.load_weights("model.h5")
	#X_val = np.load("X_train22.npy")
	#y_val = np.load("y_train22.npy")

	X_train = np.concatenate(X_train)
	y_train = np.concatenate(y_train)

	X_train, y_train = shuffle(X_train, y_train, random_state=0)

	model3.load_weights("model3.h5")
	history = model3.fit(X_train, y_train, epochs=1)
	cost.append(history.history['loss'][-1])
	model3.save_weights("model3.h5")
	#plt.plot(history.history['loss'])



plt.plot(cost)
# summarize history for loss
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Model 3'], loc='upper left')
plt.show()
