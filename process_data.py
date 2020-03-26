import glob
import time

from utils.sgf_reader import Reader

file_paths = []

path = "data/aya_self_play"

for i in range(1, 2):
    if i < 10:
        mypath = path + "/0" + str(i)
    else:
        mypath = path + "/" + str(i)

    file_paths += glob.glob(mypath + "/*.sgf")


sgfr = Reader()

sgfr.generate_data(file_paths, "data/aya_dataset/")
