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

model.load_weights("model.h5")