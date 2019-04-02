from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from SGF_Loader import get_data
from OptimizedGoTrainer import GoTrainer 
import os
from tqdm import tqdm

GT = GoTrainer()

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='tanh', padding="same", input_shape=(9,9,1)))
model.add(Conv2D(32, kernel_size=3, activation='tanh', padding="same"))
model.add(Flatten())
model.add(Dense(81, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# get data file paths
for path in os.walk("GoSampleData"):
    paths = (path[2][0:10])

print("Playing Out Games: ")
for i in tqdm(range(len(paths))):
    data, winner = get_data("GoSampleData/"+paths[i])
    GT.play(data=data, winner=winner)

print("Training Teeny Go Neural Net: ")
for i in range(1000):
    for j in range(len(GT.x_data)):

        X_train = GT.x_data[j].reshape(9, 9)
        y_train = GT.y_data[j].reshape(9, 9)
        #train the model
        model.fit(X_train, y_train, epochs=3)




