import time
from OptimizedGoTrainer import GoTrainer
import game_processor_py
import game_processor
from SGF_Loader import get_data
import os
import numpy as np

for path in os.walk("GoSampleData"):
    paths = (path[2][0:10])



GT = GoTrainer()

data, winner = get_data("GoSampleData/"+paths[0])


gt_T = time.time()
GT.play(data, winner)
gt_T = time.time() - gt_T

py_T = time.time()
x1, y1 = game_processor_py.play(data, winner)
py_T = time.time() - py_T

cy_T = time.time()
x, y = game_processor.play(data, winner)
cy_T = time.time() - cy_T

if GT.x_data == np.array(x):
    print("same")

print(np.array(GT.x_data)[2])
print(np.array(x)[2])

print(gt_T, py_T, cy_T)

print("Cython is ", py_T/cy_T,"times faster than python")
print("and ", gt_T/cy_T, "times faster then numpy")

