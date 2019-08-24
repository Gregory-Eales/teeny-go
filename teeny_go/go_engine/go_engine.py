import numpy as np



class GoEngine(self):

    def __init__(self):

        # initialize game attributes
        self.board = None
        self.turn = None
        self.white_score = None
        self.black_score = None
        self.playing = None
        self.deciding_move = True

    def new_game(self):
        self.board = self.create_board()
        self.turn = "black"
        self.white_score = 0
        self.black_score = 0
        self.playing = True
        self.deciding_move = True

    def create_board(self):
        return np.zeros([9, 9])

    def play(self):

        while self.playing:

            while self.deciding_move:

                pass
