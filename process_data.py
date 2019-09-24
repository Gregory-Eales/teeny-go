import glob

#from utils.sgf_reader import sgf_reader




files = []


path = "data/aya_self_play"

for i in range(19, 20):
    if i < 10:
        mypath = path + "/0" + str(i)
    else:
        mypath = path + "/" + str(i)

    files += glob.glob(mypath + "/*.sgf")


print(files[0])
