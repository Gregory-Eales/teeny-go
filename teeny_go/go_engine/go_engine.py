import numpy as np



class GoEngine(object):

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
        self.board_cache = None
        self.new_game()

    def new_game(self):
        self.board = self.create_board()
        self.turn = "black"
        self.white_score = 0
        self.black_score = 0
        self.end_score = 0
        self.board_cache = []

    def create_board(self):
        return np.zeros([9, 9])

    def print_board(self):
        for _ in range(9):
            print(self.board[_])

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
        self.board[move[1]][move[0]] = self.turn_to_num[self.turn]

    def get_pos_state(self, pos):
        return self.board[pos[1]][pos[0]]

    def change_turn(self):

        self.board_cache.append(self.board)

        if self.turn == "black":
            self.turn == "white"
        else:
            self.turn == "black"

    ######################
    # Game Logic Methods #
    ######################

    def move_is_valid(self, move):

        valid = False

        # check if space is empty
        if self.get_pos_state(move) != 0:
            return False

        else:
            self.make_move(move)

        # check if has liberties
        if self.has_liberties(move) == True:
            valid = True

        # if no liberties check if capturing enemy
        if self.is_capturing_enemy(move) == True:
            self.capture_enemy(move)
            valid = True

        # get group
        group = self.get_group(move)

        # check if group has liberties
        if self.check_group_liberties(group) == True:
            return True

        # and if group has liberties


        # if group


        if valid == False:
            self.board = self.board_cache[-1]
            return False

        else:
            return True

    def has_existed(self):

        for board in self.board_cache:
            if board == self.board:
                return True
        else:
            return False

    def has_liberties(self, loc):

        if loc[0] > 0 and loc[0] < 8 and loc[1] > 0 and loc[1] < 8:

            if self.get_pos_state([loc[1]-1, loc[0]]) == 0:
                return True

            if self.get_pos_state([loc[1]+1, loc[0]]) == 0:
                return True

            if self.get_pos_state([loc[1], loc[0]-1]) == 0:
                return True

            if self.get_pos_state([loc[1], loc[0]+1]) == 0:
                return True

        if loc[0] != 0:
            if self.get_pos_state([loc[1], loc[0]-1]) == 0:
                return True

        if loc[0] != 8:
            if self.get_pos_state([loc[1], loc[0]+1]) == 0:
                return True

        if loc[1] != 0:
            if self.get_pos_state([loc[1]-1, loc[0]]) == 0:
                return True

        if loc[1] != 8:
            if self.get_pos_state([loc[1]+1, loc[0]]) == 0:
                return True

        return False

    def get_near(self, loc, type):

        near = []

        if loc[0] > 0 and loc[0] < 8 and loc[1] > 0 and loc[1] < 8:

            if self.get_pos_state([loc[1]-1, loc[0]]) == type:
                near.append([loc[0], loc[1]-1])

            if self.get_pos_state([loc[1]+1, loc[0]]) == type:
                near.append([loc[0], loc[1]+1])

            if self.get_pos_state([loc[1], loc[0]-1]) == type:
                near.append([loc[0]-1, loc[1]])

            if self.get_pos_state([loc[1], loc[0]+1]) == type:
                near.append([loc[0]+1, loc[1]-1])

        if loc[0] != 0:
            if self.get_pos_state([loc[1], loc[0]-1]) == type:
                near.append([loc[0]-1, loc[1]])

        if loc[0] != 8:
            if self.get_pos_state([loc[1], loc[0]+1]) == type:
                near.append([loc[0]+1, loc[1]])

        if loc[1] != 0:
            if self.get_pos_state([loc[1]-1, loc[0]]) == type:
                near.append([loc[0], loc[1]-1])

        if loc[1] != 8:
            if self.get_pos_state([loc[1]+1, loc[0]]) == type:
                near.append([loc[0], loc[1]+1])

        if near != []:
            return near
        else:
            return False

    def is_capturing_enemy(self, loc):
        pass

    def capture_enemy(self, loc):
        pass

    def get_group(self, loc):
        type = self.board[loc[1]][loc[0]]
        group = []
        searching = True
        # if near spots = same type add to group
        near = self.get_near(loc, type)
        while searching:
            searching = False
            for space in near:





    def check_group_liberties(self, group):
        pass


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
    engine = GoEngine()
    engine.board = create_board()
    engine.print_board()
    print(engine.has_liberties([2, 0]))

if __name__ == "__main__":
    main()
