from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from SGF_Loader import get_data
from OptimizedGoTrainer import GoTrainer 
import os
from tqdm import tqdm
import numpy as np

GT = GoTrainer()

#create model
model = Sequential()
#add model layers
model.add(Conv2D(10, kernel_size=2, activation='relu', padding="same", input_shape=(9,9,1)))
model.add(Conv2D(10, kernel_size=2, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(81, activation='softmax'))
model.add(Dense(81, activation='softmax'))


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# get data file paths
for path in os.walk("GoSampleData"):
    paths = (path[2][0:1])

print("Playing Out Games: ")
for i in tqdm(range(len(paths))):
    data, winner = get_data("GoSampleData/"+paths[i])
    GT.play(data=data, winner=winner)

print("Training Teeny Go Neural Net: ")
X_train = []
y_train = []
for i in range(10000):
    for j in range(len(GT.x_data)):
    	
    	X_train.append(GT.x_data[j].reshape(9, 9, 1))
    	y_train.append(GT.y_data[j].reshape(81))
    	

X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train.shape)
print(y_train.shape)

model.fit(X_train, y_train, epochs=1)


