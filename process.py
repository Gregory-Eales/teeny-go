import glob
from os import listdir
import time
import codecs
from utils.sgf import Reader

file_paths = []

path = "./data/ogs_dan_games/"

file_paths = listdir(path)

for i in range(len(file_paths)):

    file_paths[i] = path+"/"+file_paths[i]

sgfr = Reader()

print(len(file_paths))
sgfr.generate_data(file_paths, "./data/ogs_all_dan_tensor_games/", save=True)

print("Completed:", sgfr.completed, "games")
