import GoGameViewer
import os

#path = "GoSampleData/godata1.sgf"


def get_sgf_raw_data(path):
    file = open(path)
    return file.readlines(0)

def clean_sgf_data(raw_data):
    data_holder = []
    data = []
    del(raw_data[0])
    del(raw_data[0])
    for i in range(1, len(raw_data)):
        data_holder = data_holder + raw_data[i].split(';')
    for i in data_holder:
        if i != '':
            data.append(i[:5])

    return data

def get_winner(line):
    winner = line[2]
    winner = winner.split('RE')[1][1]
    if winner == "W":
        return "white"
    if winner == "B":
        return "black"
    return winner

def translate_data(raw):
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    lower_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    data = []
    for i in raw:
        #print(i)
        if len(i) > 4:
            if i[2] in lower_letters and i[3] in lower_letters:
                data.append(i[2].upper()+str(letters.index(i[3].upper())))

    return data


def get_data(path):
    raw = get_sgf_raw_data(path)
    winner = get_winner(raw)
    raw = clean_sgf_data(raw)
    data = translate_data(raw)
    return data, winner



#print(get_winner(get_sgf_raw_data(path)))


#GG = GoGameViewer.GoEngine()

"""

for path in os.walk("GoSampleData"):
    paths = (path[2])
"""
"""
for path in paths:
    data, winner = get_data("GoSampleData/"+path)
    GG.play(data=data)
"""

#data, winner = get_data("GoSampleData/619921.sgf")
#GG.play(data=data)
