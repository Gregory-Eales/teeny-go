import numpy as np



class GoEngine(self):

    #######################
    # Initializer Methods #
    #######################

    def __init__(self):

        # initialize game attributes
        self.board = None
        self.turn = None
        self.white_score = None
        self.black_score = None
        self.end_score = None
        self.playing = None
        self.turn_to_num = {"white": -1, "black": 1}

    def new_game(self):
        self.board = self.create_board()
        self.turn = "black"
        self.white_score = 0
        self.black_score = 0
        self.end_score = 0

    def create_board(self):
        return np.zeros([9, 9])

    #######################
    # Game Action Methods #
    #######################

    def play(self):

        self.new_game()

        while self.playing:
            # get move
            move = self.get_move()
            # check if move is valid
            if self.move_is_valid(move) == True:
                # if valid, make move
                self.make_move(move)
                #   change turn
                self.change_turn()

        self.score_game()

    def get_move(self):
        pass

    def make_move(self, move):
        self.board[move[0]][move[1]] = self.turn_to_num[self.turn]

    def change_turn(self):
        if self.turn == "black":
            self.turn == "white"
        else:
            self.turn == "black"
