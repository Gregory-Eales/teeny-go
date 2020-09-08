import glob
from os import listdir
import time
import codecs
from utils.ogs_sgf_reader import Reader

file_paths = []

path = "./data/ogs_dan_games/"

file_paths = listdir(path)

for i in range(len(file_paths)):

    file_paths[i] = path+"/"+file_paths[i]

sgfr = Reader()

print(len(file_paths))
sgfr.generate_data(file_paths[0:10], "./data/new_ogs_tensor_games/", save=False)

print("Completed: ", sgfr.completed)
