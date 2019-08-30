import numpy as np
cimport numpy as np
import copy


cdef class GoEngine():

    #######################
    # Initializer Methods #
    #######################

    # initialize game attributes
    cdef public np.ndarray board
    cdef public str turn
    cpdef public int white_score
    cpdef public int black_score
    cdef int black_holder
    cdef int white_holder
    cdef int end_score
    cpdef public bint is_playing
    cpdef public bint is_deciding
    cpdef int pass_count
    cpdef dict turn_to_num
    cdef list board_cache

    def __init__(self):
        turn = "black"
        white_score = 0
        black_score = 0
        black_holder = 0
        white_holder = 0
        end_score = 0
        is_playing = True
        is_deciding = True
        pass_count = 0
        turn_to_num = {"white": -1, "black": 1}
        board_cache = []
        self.new_game()


    cpdef new_game(self):
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
        self.turn_to_num = turn_to_num = {"white": -1, "black": 1}
        cdef int i
        for i in range(5):
            self.board_cache.append(copy.deepcopy(self.board))

    cdef np.ndarray create_board(self):
        return np.zeros([9, 9], dtype=np.int)

    cpdef print_board(self):
        cdef int _
        for _ in range(9):
            print(self.board[_])

    #######################
    # Game Action Methods #
    #######################

    cpdef make_move(self, py_move):
        cdef list move = py_move
        self.board[move[1]][move[0]] = self.turn_to_num[self.turn]

    cdef int get_pos_state(self, py_pos):
        cdef list pos = py_pos
        return self.board[pos[1]][pos[0]]

    cpdef change_turn(self):
        if self.turn == "black":
            self.turn = "white"
        elif self.turn == "white":
            self.turn = "black"


    ######################
    # Game Logic Methods #
    ######################

    cpdef bint check_valid(self, py_move):
        cdef move = py_move
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


        cdef bint valid = False

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
        cdef group
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

        self.capture_all_pieces()

        if valid == False or self.has_existed() == True:
            self.board = copy.deepcopy(self.board_cache[-1])
            self.black_holder = 0
            self.white_holder = 0
            return False

        else:
            self.change_turn()
            #self.make_move(move)
            self.is_deciding = False
            self.black_score += self.black_holder
            self.white_score += self.white_holder
            self.black_holder = 0
            self.white_holder = 0
            self.board_cache.append(copy.deepcopy(self.board))
            #if np.sum(np.where(self.board != 0, 1, 0)) > 75:
                #self.is_playing = False
            return True

    cdef capture_all_pieces(self):
        cdef int i
        cdef int j
        cdef list group
        for j in range(9):
            for i in range(9):
                group = self.get_group([i, j])
                if group != False:
                    if self.board[group[0][1]][group[0][0]] != self.turn_to_num[self.turn]:
                        if self.check_group_liberties(group)==False:
                            self.capture_group(group)

    cdef bint has_existed(self):
        cdef np.ndarray board
        for board in self.board_cache:
            if np.array_equal(self.board, board):
                return True
        else:
            return False

    cdef bint has_liberties(self, py_loc):
        cdef list loc = py_loc

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

    cdef list get_near(self, py_loc, py_type):

        cdef list loc = py_loc
        cdef int type = py_type

        cdef list near = []

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
            return near

    cdef bint is_capturing_enemy(self):

        cdef int i
        cdef int j
        cdef list group

        for j in range(9):
            for i in range(9):
                group = self.get_group([i, j])
                if group != False:
                    if self.board[group[0][1]][group[0][0]] != self.turn_to_num[self.turn]:
                        if self.check_group_liberties(group)==False:
                            return True
        return False

    cdef capture_group(self, group):
        cdef list loc
        for loc in group:
            self.capture_piece(loc)

    cdef capture_piece(self, py_loc):
        cdef list loc = py_loc
        if self.turn == "white":
            self.white_holder += 1
        else:
            self.black_holder += 1
        self.board[loc[1]][loc[0]] = 0

    cdef list get_group(self, py_loc):
        cdef list loc = py_loc
        cdef int type = self.board[loc[1]][loc[0]]
        cdef list group
        cdef bint searching
        cdef list near
        cdef list space
        cdef list n

        if type == 0: return False
        group = [loc]
        searching = True
        near = self.get_near(loc, type)
        if near == []: pass

        else:
            group = group + near
        while searching:
            searching = False
            for space in group:
                near = self.get_near(space, type)
                if near != []:
                    for n in near:
                        if n not in group:
                            searching = True
                            group.append(n)
        return group


    cdef bint check_group_liberties(self, group):

        cdef list space

        for space in group:
            if self.has_liberties(space) == True:
                return True
        return False

    cdef get_all_groups(self):
        pass


    cpdef str score_game(self):

        self.white_score += np.sum(np.where(self.board==-1, 1, 0))
        self.black_score += np.sum(np.where(self.board==1, 1, 0))

        if self.black_score > self.white_score:
            return "black"

        else:
            return "white"

    cpdef np.ndarray get_board_tensor(self):
        cdef list black = []
        cdef white = []
        cdef list turn = []
        cdef int i


        if self.turn == "white":
            turn = [np.zeros([9, 9])]
        else:
            turn = [np.ones([9, 9])]

        for i in range(1, 6):
            black.append(np.where(self.board_cache[-i] == 1, 1, 0))
            white.append(np.where(self.board_cache[-i] == -1, 1, 0))

        return np.array(black+white+turn).reshape([1, 11, 9, 9])
