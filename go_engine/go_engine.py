
class GoEngine(object):

    def __init__(self):

        # initialize game states
        self.board = self.initialize_board()
        self.turn = "black"
        self.playing = True
        self.making_move = True
        self.move = [0, 0]
        self.turn_number = {"white":-1, "black"=1}

    def play(self):

        while self.playing:

            # show board

            while self.making_move:

                # get move

                # check if move is valid
                if check_valid() == True:
                    # make move
                    self.making_move()
                    self.making_move = False



            # change turn
            self.change_turn()
            self.making_move = True
            pass

    def make_move(self):
        self.board[self.move[1]][self.board[0]] = self.turn_number[self.turn]

    def change_turn(self):

        if self.turn == "black":
            turn = "white"

        else:
            self.turn = "black"

    def initialize_board():
        board = []
        spaces = []
        for i in range(9):
            spaces.append(0)

        for i in range(9):
            board.append(spaces)

        return board

    def check_valid(self):

        # check if space is occupied
        if self.board[self.move[1]][self.board[0]] != 0:
            return False

        # check to see if liberties are free
        if self.has_liberties() == False:
            return False

        # check if group is killed
        if self.killing_group() == False:
            return False


    def has_liberties(self):
        pass

    def killing_group(self):
        pass




def main():
    pass

if __name__ == "__main__":
    main()
