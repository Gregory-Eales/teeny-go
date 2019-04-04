from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from SGF_Loader import get_data
from OptimizedGoTrainer import GoTrainer 
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

GT = GoTrainer()

#create model
model = Sequential()
#add model layers
model.add(Conv2D(60, kernel_size=2, activation='tanh', padding="same", input_shape=(9,9,1)))
model.add(Conv2D(50, kernel_size=2, activation='tanh', padding="same"))
model.add(Conv2D(30, kernel_size=2, activation='tanh', padding="same"))

#model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(160, activation='tanh', use_bias=True))
model.add(Dense(81, activation='sigmoid', use_bias=True))


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# get data file paths
for path in os.walk("GoSampleData"):
    paths = (path[2][11:21])

print("Playing Out Games: ")
for i in tqdm(range(len(paths))):
    data, winner = get_data("GoSampleData/"+paths[i])
    GT.play(data=data, winner=winner)

print("Training Teeny Go Neural Net: ")
X_train = []
y_train = []
for i in range(10):
    for j in range(len(GT.x_data)):
    	
    	X_train.append(GT.x_data[j].reshape(9, 9, 1))
    	y_train.append(GT.y_data[j].reshape(81))
    	

X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)

#model.load_weights("model.h5")
model.load_weights("model.h5")
history = model.fit(X_train, y_train, epochs=15)
model.save_weights("model.h5")
plt.plot(history.history['loss'])

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


