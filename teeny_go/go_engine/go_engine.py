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
        self.last_taken_white = None
        self.last_taken_black = None
        self.taken_counter = 0



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

        return np.zeros([9, 9])
        return board

    def check_valid(self, move):

        valid = True

        # check if space is occupied
        if self.board[move[1]][move[0]] != 0:
            print("Space is Occupied")
            return False

        self.board[move[1]][move[0]] = self.turn_number[self.turn]

        # check to see if liberties are free
        if self.check_individual_lib(move) == False:
            print("No Open liberties")
            valid = False

        group = self.get_group(move, self.turn_number[self.turn])

        if group != []:
            # check if group is killed
            print("checking group")
            if self.is_killing_group(group) == False:
                print("Not killing same group")
                valid = True


            if self.killing_enemy_group(move) == True:
                print("killing enemy")
                valid=True

        if valid == False:
            self.board[move[1]][move[0]] = 0

        return valid

    def capture_all_pieces(self):
        type = self.turn_number[self.turn]*-1
        print("killing Pieces")
        for i in range(9):
            for j in range(9):
                if self.board[j][i] == type:
                    print("same board type:", type)
                    group = self.get_group([i, j], type)
                    if self.check_group_liberties(group) == False:
                        self.capture_group(group)


    def check_individual_lib(self, move):
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

        enemies = self.get_near(move, type)


        return False

    def get_near(self, move, type):

        near = []

        if move[0] > 0 and move[0] < 8 and move[1] > 0 and move[1] < 8:
            if self.board[move[1]][move[0]+1] == type:
                near.append([move[0]+1, move[1]])


            if self.board[move[1]][move[0]-1] == type:
                near.append([move[0]-1, move[1]])


            if self.board[move[1]+1][move[0]] == type:
                near.append([move[0], move[1]+1])


            if self.board[move[1]-1][move[0]] == type:
                near.append([move[0], move[1]-1])

        else:

            if move[0] != 8:
                if self.board[move[1]][move[0]+1] == type:
                    near.append([move[0]+1, move[1]])

            if move[0] != 0:
                if self.board[move[1]][move[0]-1] == type:
                    near.append([move[0]-1 ,move[1]])

            if move[1] != 8:
                if self.board[move[1]+1][move[0]] == type:
                    near.append([move[0], move[1]+1])

            if move[1] != 0:
                if self.board[move[1]-1][move[0]] == type:
                    near.append([move[0], move[1]-1])

        near.append(move)
        if near != []:
            if move not in near and self.board[move[1]][move[0]] == type:
                near.append(move)
            return near

        else:
            return False

    def check_group_liberties(self, group):

        for space in group:
            if self.check_individual_lib(space) == True:
                return True

        return False


    def is_killing_group(self, group):

        if self.check_group_liberties(group) == True:
            return False

        return True

    def capture_group(self, group):
        for space in group:
            self.capture_piece(space)

    def capture_piece(self, pos):
        print("x, y: ", pos[1], pos[0])
        self.board[pos[1]][pos[0]] = 0

    def get_group(self, loc, type):

        group = []
        near = self.get_near(loc, type)
        if near != False:
            group += near
        for i in range(1):
            for elmn in group:
                near = self.get_near(elmn, type)
                if near != False:
                    for elmn in near:
                        if elmn not in group:
                            group.append(elmn)
        return group

    def score_board(self):
        pass

    def killing_enemy_group(self, loc):

        capture = True
        enemy_type = self.turn_number[self.turn]*-1
        near = self.get_near(loc, enemy_type)
        enemy_groups = []
        if near != False:
            for elmnt in near:
                group = self.get_group(elmnt, enemy_type)
                if group != False:
                    enemy_groups.append(group)

        for g in enemy_groups:
            if self.check_group_liberties(g) == False:
                print("Capturinng")
                print("groupy", g)

                if self.turn == "white":

                    if len(g) == 1:
                        print("capturing single")
                        if self.last_taken_white == g[0]:
                            capture = False
                        else:
                            self.last_taken_white = g[0]
                            self.taken_counter += 1

                if self.turn == "black":

                    if len(g) == 1:
                        print("capturing single")
                        if self.last_taken_black == g[0]:
                            capture = False
                        else:
                            self.last_taken_black = g[0]
                            self.taken_counter += 1

                if self.taken_counter > 4:
                    self.last_taken_black = []
                    self.last_taken_white = []
                    self.taken_counter = 0

                if capture == True:
                    self.taken_counter+=1
                    self.capture_group(g)
                    return True
                else:
                    return False


def create_board():
    board = []
    for i in range(3):
        board.append([1, 1, 1, 1, 1, 1, 1, 1, 1])
    for i in range(1):
        board.append([0, 0, 0, 0, 0, 0, 0, 0, 0])

    board.append([0, 0, 0 ,0 ,0 ,0, 0, 0, 1])
    for i in range(4):
        board.append([0, 0, 0, 0, 0, 0, 0, 0, 0])

    board[1][5] = -1
    return np.zeros([9,9])
    return board




def main():
    go = GoEngine()
    go.board = create_board()
    go.print_board()
    group = go.get_group([0, 0], 1)
    print(len(group))
    go.capture_group(group)
    go.print_board()

if __name__ == "__main__":
    main()
