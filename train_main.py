from TrainTeenyGoAI import GoTrainingEngine
import numpy as np
import os


trainer = GoTrainingEngine()
#trainer.NN.load_weights()

for path in os.walk("GoSampleData"):
    paths = (path[2])

#trainer.play(paths[10])


for i in range(1):
    trainer.play(paths[0])

trainer.NN.save_weights()
