import glob
import time

from utils.sgf_reader import Reader

file_paths = []

path = "data/aya_sgf"

for i in range(1, 21):
    if i < 10:
        mypath = path + "/0" + str(i)
    else:
        mypath = path + "/" + str(i)

    file_paths += glob.glob(mypath + "/*.sgf")


sgfr = Reader()

sgfr.generate_data(file_paths, "data/aya_dataset/")
