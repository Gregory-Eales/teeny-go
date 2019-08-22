import numpy as np


class GoEngine(object):

    def __init__(self):

        # initialize game states
        self.board = self.initialize_board()
        self.turn = "black"
        self.playing = True
        self.making_move = True
        self.move = [0, 0]
        self.turn_number = {"white":-1, "black":1}
        self.black_score, self.white_score = self.initialize_score()


    # check if space is empty
    # check to see if same pieces near
    # get group of near pieces
    # check liberties of goup
    # if no liberties check adjacent black pieces for capture
    # if unable to capture return  false



    def play(self):

        #while self.playing:
        for i in range(10):

            # show board
            self.print_board()
            print("Turn:", self.turn)

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
                    print("coordinate must  0 through 8")
                else:
                    deciding = False
            except:
                print("coordinate must be an integer")

        self.move = [x, y]

    def make_move(self):
        self.board[self.move[1]][self.move[0]] = self.turn_number[self.turn]

    def change_turn(self):
        print("changed turn")
        if self.turn == "black":
            self.turn = "white"
        else:
            self.turn = "black"

    def initialize_score(self):
        return 0, 0

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
        if self.has_liberties() == True:
            return True

        group = self.get_group(self.move)

        # check if group is killed
        if self.is_killing_group(group) == False:
            return True

        if killing_enemy_group(self.group) == True:
            return True

        return True

    def check_individual_lib(self, move):
        print("Move: ", move)
        if move[0] > 0 and move[0] < 8 and move[1] > 0 and move[1] < 8:
            if self.board[move[1]][move[0]+1] == 0:
                return True


            if self.board[move[1]][move[0]-1] == 0:
                return True


            if self.board[move[1]+1][move[0]] == 0:
                return True


            if self.board[move[1]-1][move[0]] == 0:
                return True

        else:

            if move[0] != 8:
                if self.board[move[1]][move[0]+1] == 0:
                    return True

            if move[0] != 0:
                if self.board[move[1]][move[0]-1] == 0:
                    return True

            if move[1] != 8:
                if self.board[move[1]+1][move[0]] == 0:
                    return True

            if move[1] != 0:
                if self.board[move[1]-1][move[0]] == 0:
                    return True

        return False

    def check_enemy_near(self, move):

        type = self.board[move[1]][move[0]]*-1

        if type==0:
            return False

        if move[0] > 0 and move[0] < 8 and move[1] > 0 and move[1] < 8:
            if self.board[move[1]][move[0]+1] == type:
                [move[1],move[0]+1]


            if self.board[move[1]][move[0]-1] == type:
                [move[1],move[0]-1]


            if self.board[move[1]+1][move[0]] == type:
                [move[1]+1,move[0]]


            if self.board[move[1]-1][move[0]] == type:
                [move[1]-1,move[0]]

        else:

            if move[0] != 8:
                if self.board[move[1]][move[0]+1] == type:
                    near.append([move[1],move[0]+1])

            if move[0] != 0:
                if self.board[move[1]][move[0]-1] == type:
                    near.append([move[1],move[0]-1])

            if move[1] != 8:
                if self.board[move[1]+1][move[0]] == type:
                    near.append([move[1]+1, move[0]])

            if move[1] != 0:
                if self.board[move[1]-1][move[0]] == type:
                    near.append([move[1]-1, move[0]])


    def get_near(self, move):
        type = self.board[move[1]][move[0]]

        if type == 0:
            return False

        near = []


        if move[0] > 0 and move[0] < 8 and move[1] > 0 and move[1] < 8:
            if self.board[move[1]][move[0]+1] == type:
                if [move[1], move[0]+1] not in near:
                    near.append([move[1],move[0]+1])


            if self.board[move[1]][move[0]-1] == type:
                if [move[1], move[0]-1] not in near:
                    near.append([move[1],move[0]-1])


            if self.board[move[1]+1][move[0]] == type:
                if [move[1]+1, move[0]] not in near:
                    near.append([move[1]+1,move[0]])


            if self.board[move[1]-1][move[0]] == type:
                if [move[1]-1, move[0]] not in near:
                    near.append([move[1]-1,move[0]])

        else:

            if move[0] != 8:
                if self.board[move[1]][move[0]+1] == type:
                    if [move[1], move[0]+1] not in near:
                        near.append([move[1],move[0]+1])

            if move[0] != 0:
                if self.board[move[1]][move[0]-1] == type:
                    if [move[1], move[0]-1] not in near:
                        near.append([move[1],move[0]-1])

            if move[1] != 8:
                if self.board[move[1]+1][move[0]] == type:
                    if [move[1]+1, move[0]] not in near:
                        near.append([move[1]+1, move[0]])

            if move[1] != 0:
                if self.board[move[1]-1][move[0]] == type:
                    if [move[1]-1, move[0]] not in near:
                        near.append([move[1]-1, move[0]])

        if near != []:
            if move not in near:
                near.append(move)
            return near

        else:
            return False

    def check_group_liberties(self, group):
        print("group:", group)
        for space in group:

            if self.check_individual_lib(space) == True:
                return True

        return False


    def is_killing_group(self, loc):

        group = self.get_group(loc)

        if self.check_group_liberties(group) == False:
            return True

        return False

    def capture_group(self, group):
        for space in group:
            self.capture_piece(space)

    def capture_piece(self, pos):
        self.board[pos[1]][pos[0]] = 0

    def get_group(self, loc):

        group = [loc]

        for i in range(10):
            for elmn in group:
                near = self.get_near(elmn)
            if near != False:
                for elmn in near:
                    if elmn not in group:
                        group.append(elmn)
        return group

def create_board():
    board = []

    for i in range(3):
        row = []
        for j in range(9):
            row.append(1)
        board.append(row)

    for i in range(6):
        row = []
        for j in range(9):
            row.append(0)
        board.append(row)

    return board


def main():
    go = GoEngine()
    go.board = create_board()
    go.print_board()
    group = go.get_group([0, 0])
    print(len(group))

if __name__ == "__main__":
    main()
