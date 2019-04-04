from OptimizedGoTrainer import GoTrainer
from GoGameViewer import GoViewer
from SGF_Loader import get_data
from GoGraphics import GoEngine


GT = GoTrainer()
GV = GoViewer()

path = "GoSampleData/619922.sgf"

data, winner = get_data(path)

#GT.play(data=data)
#print(GT.board)

for i in range(15):
	GV.play(data)
"""

go = GoEngine()
go.play()
"""