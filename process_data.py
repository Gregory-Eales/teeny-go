import glob
from os import listdir
import time

from utils.ogs_sgf_reader import Reader

file_paths = []

path = "./data/ogs_dan_games/"

file_paths = listdir(path)
file_paths = ["19946992-027-minusGo-Rosequartz.sgf"]

for i in range(len(file_paths)):
    file_paths[i] = path+"/"+file_paths[i]

print(len(file_paths))



"""
for i in range(1, 2):
    if i < 10:
        mypath = path + "/0" + str(i)
    else:
        mypath = path + "/" + str(i)

    file_paths += glob.glob(mypath + "/*.sgf")
"""

sgfr = Reader()

sgfr.generate_data(file_paths, "./data/ogs_dan_games/")
