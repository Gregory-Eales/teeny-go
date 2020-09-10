
import sys

from . import gtp
from . import go

class Engine:
    name = 'Disco'

    def __init__(self):
        self.board = go.Board()

    def boardsize(self, size):
        if int(size) != go.SIZE:
            raise 'illegal board size'

    def clear_board(self):
        self.board = go.Board()

    def komi(self, value):
        pass

    def play(self, color, vertex):
        vertex = gtp.parse_vertex(vertex)
        if vertex is None:
            self.board.move(go.PASS)
        else:
            i, j = vertex
            pos = go.to_pos(i, j)
            self.board.move(pos)

#    def undo(self):
#        pass

    def genmove(self, color):
        pos = go.computer_move(self.board)
        self.board.move(pos)
        if pos == go.PASS:
            return 'pass'
        x, y = go.to_xy(pos)
        gtp.make_vertex(x, y)

        return x + y*9

#    def showboard(self):
#        pass

def main():
    engine = Engine()
    gtp.run(engine)
    print("done")

if __name__ == '__main__':
    main()
