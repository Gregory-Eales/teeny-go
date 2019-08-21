
class GoEngine(object):

    def __init__(self):

        # initialize game states
        self.board = self.initialize_board()
        self.turn = "black"
        self.playing = True
        self.making_move = True
        self.move = [0, 0]
        self.turn_number = {"white":-1, "black":1}

    def play(self):

        while self.playing:

            # show board
            self.print_board()

            while self.making_move:

                # get move
                self.get_move()

                # check if move is valid
                if self.check_valid() == True:
                    
                    # make move
                    self.make_move()
                    self.making_move = False

            # change turn
            self.change_turn()
            self.making_move = True


    def print_board(self):
        for i in range(9):
            print(i, self.board[i])


    def get_move(self):
        deciding = True
        while deciding:
            y = input("y-coordinate: ")
            x = input("x-coordinate: ")
            try:
                x = int(x)
                y = int(y)
                if x < 0 or x > 8 or y < 0 or y > 8:
                    x + "a"
                    print("coordinate must  0 through 8")
                else:
                    deciding = False
            except:
                print("coordinate must be an integer")

        self.move = [y, x]

    def make_move(self):
        self.board[self.move[1]][self.move[0]] = self.turn_number[self.turn]

    def change_turn(self):

        if self.turn == "black":
            turn = "white"

        else:
            self.turn = "black"

    def initialize_board(self):
        board = []
        for i in range(9):
            board.append([0, 0, 0, 0, 0, 0, 0, 0, 0])

        return board

    def check_valid(self):

        # check if space is occupied
        if self.board[self.move[1]][self.move[0]] != 0:
            return False

        # check to see if liberties are free
        if self.has_liberties() == False:
            return False

        # check if group is killed
        if self.killing_group() == False:
            return False

        return True


    def has_liberties(self):
        return True

    def killing_group(self):
        return True




def main():
    go = GoEngine()
    go.play()

if __name__ == "__main__":
    main()
