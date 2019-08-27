import numpy as np
import copy



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
        self.black_holder = None
        self.white_holder = None
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
        self.black_holder = 0
        self.white_holder = 0
        self.end_score = 0
        self.board_cache = []
        for i in range(5):
            self.board_cache.append(copy.deepcopy(self.board))

    def create_board(self):
        return np.zeros([9, 9], dtype=np.int)

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
        if self.turn == "black":
            self.turn = "white"
        elif self.turn == "white":
            self.turn = "black"


    ######################
    # Game Logic Methods #
    ######################

    def check_valid(self, move):

        valid = False

        # check if space is empty
        if self.get_pos_state(move) != 0:
            return False

        else:
            self.make_move(move)

        # check if has liberties
        if self.has_liberties(move) == True:
            valid = True
        else:
            # if no liberties check if capturing enemy
            if self.is_capturing_enemy() == True:
                valid = True
            else:
                valid = False

        # get group
        group = self.get_group(move)

        # check if group has liberties
        if group != False:
            if self.check_group_liberties(group) == True:
                valid = True

            else:
                # if no liberties check if capturing enemy
                if self.is_capturing_enemy() == True:
                    valid = True
                else:
                    valid = False


        # and if group has liberties

        # if group
        self.capture_all_pieces()

        if valid == False or self.has_existed() == True:
            self.board = copy.deepcopy(self.board_cache[-1])
            self.black_holder = 0
            self.white_holder = 0
            return False

        else:
            self.black_score += self.black_holder
            self.white_score += self.white_holder
            self.black_holder = 0
            self.white_holder = 0
            self.board_cache.append(copy.deepcopy(self.board))
            return True

    def capture_all_pieces(self):
        for j in range(9):
            for i in range(9):
                group = self.get_group([i, j])
                if group != False:
                    if self.board[group[0][1]][group[0][0]] != self.turn_to_num[self.turn]:
                        if self.check_group_liberties(group)==False:
                            self.capture_group(group)

    def has_existed(self):
        for board in self.board_cache:
            if np.array_equal(self.board, board):
                return True
        else:
            return False

    def has_liberties(self, loc):

        if loc[0] > 0 and loc[0] < 8 and loc[1] > 0 and loc[1] < 8:

            if self.get_pos_state([loc[0], loc[1]-1]) == 0:
                return True

            if self.get_pos_state([loc[0], loc[1]+1]) == 0:
                return True

            if self.get_pos_state([loc[0]-1, loc[1]]) == 0:
                return True

            if self.get_pos_state([loc[0]+1, loc[1]]) == 0:
                return True

        if loc[0] != 0:
            if self.get_pos_state([loc[0]-1, loc[1]]) == 0:
                return True

        if loc[0] != 8:
            if self.get_pos_state([loc[0]+1, loc[1]]) == 0:
                return True

        if loc[1] != 0:
            if self.get_pos_state([loc[0], loc[1]-1]) == 0:
                return True

        if loc[1] != 8:
            if self.get_pos_state([loc[0], loc[1]+1]) == 0:
                return True

        return False

    def get_near(self, loc, type):

        near = []

        if loc[0] > 0 and loc[0] < 8 and loc[1] > 0 and loc[1] < 8:

            if self.get_pos_state([loc[0], loc[1]-1]) == type:
                near.append([loc[0], loc[1]-1])

            if self.get_pos_state([loc[0], loc[1]+1]) == type:
                near.append([loc[0], loc[1]+1])

            if self.get_pos_state([loc[0]-1, loc[1]]) == type:
                near.append([loc[0]-1, loc[1]])

            if self.get_pos_state([loc[0]+1, loc[1]]) == type:
                near.append([loc[0]+1, loc[1]])

        if loc[0] != 0:
            if self.get_pos_state([loc[0]-1, loc[1]]) == type:
                near.append([loc[0]-1, loc[1]])

        if loc[0] != 8:
            if self.get_pos_state([loc[0]+1, loc[1]]) == type:
                near.append([loc[0]+1, loc[1]])

        if loc[1] != 0:
            if self.get_pos_state([loc[0], loc[1]-1]) == type:
                near.append([loc[0], loc[1]-1])

        if loc[1] != 8:
            if self.get_pos_state([loc[0], loc[1]+1]) == type:
                near.append([loc[0], loc[1]+1])

        if near != []:
            return near
        else:
            return False

    def is_capturing_enemy(self):
        for j in range(9):
            for i in range(9):
                group = self.get_group([i, j])
                if group != False:
                    if self.board[group[0][1]][group[0][0]] != self.turn_to_num[self.turn]:
                        if self.check_group_liberties(group)==False:
                            return True
        return False

    def capture_group(self, group):
        for loc in group:
            self.capture_piece(loc)

    def capture_piece(self, loc):
        if self.turn == "white":
            self.white_holder += 1
        else:
            self.black_holder += 1
        self.board[loc[1]][loc[0]] = 0

    def get_group(self, loc):
        type = self.board[loc[1]][loc[0]]
        if type == 0: return False
        group = [loc]
        searching = True
        near = self.get_near(loc, type)
        if near == False: pass

        else:
            group = group + near
        while searching:
            searching = False
            for space in group:
                near = self.get_near(space, type)
                if near != False:
                    for n in near:
                        if n not in group:
                            searching = True
                            group.append(n)
        return group


    def check_group_liberties(self, group):
        for space in group:
            if self.has_liberties(space) == True:
                return True
        return False

    def score_game(self):
        pass

    def get_board_tensor(self):
        black = []
        white = []
        turn = None

        if self.turn == "white":
            turn = [np.zeros([9, 9])]
        else:
            turn = [np.ones([9, 9])]

        for i in range(1, 6):
            black.append(np.where(self.board_cache[-i] == 1, 1, 0))
            white.append(np.where(self.board_cache[-i] == -1, 1, 0))

        print(turn)
        return np.array(black+white+turn)

def create_board():
    board = []
    for i in range(3):
        row = []
        for j in range(9):
            row.append(1)
        board.append(row)

    board.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
    board.append([1, 0, 1, 0, 0, 1, 0, 1, 1])
    board.append([0, 0, 1, 1, 1, 1, 1, 0, 1])

    for i in range(3):
        row = []
        for j in range(9):
            row.append(0)
        board.append(row)

    return board

def main():
    engine = GoEngine()
    engine.board = create_board()
    engine.print_board()
    print("#########################")
    group = engine.get_group([0, 0])
    print(engine.black_score)
    engine.capture_group(group)
    engine.print_board()
    print(engine.black_score)

if __name__ == "__main__":
    main()
