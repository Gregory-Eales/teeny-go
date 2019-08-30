import numpy as np
import copy
import time


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
        self.is_playing = None
        self.is_deciding = None
        self.pass_count = None
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
        self.pass_count = 0
        self.end_score = 0
        self.board_cache = []
        self.is_playing = True
        self.is_deciding = True
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

        while self.is_playing:
            # get move
            move = self.get_move()

            t = time.time()
            # check if move is valid
            if self.check_valid(move) == True:
                # if valid, make move
                self.make_move(move)
                #   change turn
                self.change_turn()
            print("Game Logic Time:", (time.time()-t)*1000, "ms")

        self.score_game()

    def get_move(self):
        x = int(input("X:"))
        y = int(input("Y:"))
        return [x, y]


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
        self.is_deciding = True

        if move == "pass":
            self.change_turn()
            self.pass_count+=1
            if self.pass_count >= 2:
                self.is_playing = False
            self.is_deciding = False
            self.black_holder = 0
            self.white_holder = 0
            return True


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
                print("is capturing enemy")
                valid = True
            else:
                valid = False

        # get group
        group = self.get_group(move)

        # check if group has liberties
        if group != False:
            if self.check_group_liberties(group) == True:
                print("Has group liberties")
                valid = True

            else:
                # if no liberties check if capturing enemy
                if self.is_capturing_enemy() == True:
                    valid = True
                    print("Captured")
                else:
                    valid = False

        self.capture_all_pieces()

        if valid == False or self.has_existed() == True:
            self.board = copy.deepcopy(self.board_cache[-1])
            self.black_holder = 0
            self.white_holder = 0
            print("move invalid")
            return False

        else:
            print("Move is valid")
            self.change_turn()

            self.is_deciding = False
            self.black_score += self.black_holder
            self.white_score += self.white_holder
            self.black_holder = 0
            self.white_holder = 0
            self.board_cache.append(copy.deepcopy(self.board))
            #if np.sum(np.where(self.board != 0, 1, 0)) > 75:
                #self.is_playing = False
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
        if near == False: return group

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

    def get_all_groups(self):
        pass

    def score_game(self):

        self.white_score += np.sum(np.where(self.board==-1, 1, 0))
        self.black_score += np.sum(np.where(self.board==1, 1, 0))

        if self.black_score > self.white_score:
            return "black"

        else:
            return "white"

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

        return np.array(black+white+turn).reshape([1, 11, 9, 9])



def main():
    engine = GoEngine()
    engine.play()

if __name__ == "__main__":
    main()
