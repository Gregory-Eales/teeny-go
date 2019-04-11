import copy


cdef initialize_board():
            cdef int i
            board = []
            for i in range(9):
                board.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
            return board


# returns an empty board with the move about to be made
cdef position_to_coordinates(str move):
            cdef list letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
            return [letters.index(move[0]), int(move[1])]


cdef get_y(list boardy, list move):
            cdef int i
            cdef int j
            cdef list y
            y = copy.deepcopy(boardy)
            for i in range(9):
                for j in range(9):
                    if y[j][i] == 0:
                        y[j][i] = 0.2

            for i in range(9):
                for j in range(9):
                    if y[j][i] != 0.2:
                        y[j][i] = 0

            y[move[0]][move[1]] = 1
            return y


cpdef play(data, winner):
            cdef str turn = "black"
            cdef list board = initialize_board()
            cdef list x_data = []
            cdef list y_data = []
            cdef list boardy
            cdef int stand_off
            cdef int white_score = 0
            cdef int black_score = 0
            cdef int i
            cdef int j



            for move in data:

                type_for_capture = 0
                move = position_to_coordinates(move)
                x = copy.deepcopy(board)
                boardy = copy.deepcopy(board)
                y = (get_y(boardy, move))

                if board[move[0]][move[1]] == 0:
                    # stand off is set to False

                    stand_off = 0
                    #if the position is good then...
                    if move is not None:
                        move_state = board[move[0]][move[1]]
                        # if the board space is empty...
                        if move_state == 0:
                            # place white or black depending on the turn
                            if turn == "white":
                                board[move[0]][move[1]] = 1
                            elif turn == "black":
                                board[move[0]][move[1]] = -1
                            # if
                            check_captures = check_capture_pieces(move, board)

                            if check_captures == "white" or check_captures == "black":
                                if check_captures == "white":
                                    type_for_capture = 1
                                    stand_off = 1

                                if check_captures == "black":
                                    type_for_capture = -1
                                    stand_off = 1

                            if check_captures == 0 or stand_off == 1:
                                if turn == "white":
                                    turn = "black"
                                elif turn == "black":
                                    turn = "white"

                            if True:
                                if turn == winner:

                                    if winner == "white":
                                        x_data.append(copy.deepcopy(x))
                                        y_data.append(copy.deepcopy(y))



                                    if winner == "black":

                                        for i in range(9):
                                            for j in range(9):
                                                if x[j][i] == 1:
                                                    x[j][i] = -2

                                        for i in range(9):
                                            for j in range(9):
                                                if x[j][i] == -1:
                                                    x[j][i] = 1

                                        for i in range(9):
                                            for j in range(9):
                                                if x[j][i] == -2:
                                                    x[j][i] = -1

                                        x_data.append(copy.deepcopy(x))
                                        y_data.append(copy.deepcopy(y))




                            if check_captures == 1:
                                board[move[0]][move[1]] = 0

                if type_for_capture != 0:
                    board = capture_pieces(type_for_capture, board, white_score, black_score)
            return x_data, y_data


cdef capture_pieces(type_for_capture, board, white_score, black_score):
            cdef int i
            cdef int j

            for i in range(9):
                for j in range(9):
                    location_state = board[i][j]
                    if location_state != 0 and location_state == type_for_capture:
                        group = get_group([i, j], location_state, board)
                        if group != []:
                            free = check_neighbors(group, location_state, board)
                            if free == "False":
                                board = remove_group(group, white_score, black_score, board)
            return board


cdef check_capture_pieces(position, board):
            cdef int i
            cdef int j
            killing_itself = 0
            for i in range(9):
                for j in range(9):
                    location_state = board[i][j]
                    if location_state != 0:
                        group = get_group([i, j], location_state, board)
                        if group != []:
                            free = check_neighbors(group, location_state, board)
                            if free == "False":
                                    if position in group:
                                        killing_itself = 1
                                    if location_state == 1 and board[position[0]][position[1]] != 1:
                                        return "white"
                                    if location_state == -1 and board[position[0]][position[1]] != -1:
                                        return "black"

            return killing_itself


cdef check_neighbors(list group, int state_type, list board):
            cdef str liberty = "False"

            for position in group:

                a, b = position[0], position[1]

                if a < 8:
                    if board[a+1][b] == 0:
                        return True

                if a > 0:
                    if board[a-1][b] == 0:
                        return True

                if b < 8:
                    if board[a][b+1] == 0:
                        return True

                if b > 0:
                    if board[a][b-1] == 0:
                        return True

            return liberty


cdef get_group(position, state_type, board):
            cdef list stone_group = []
            stone_group.append(position)
            cdef int i
            for j in range(20):
                for pos in stone_group:
                    a, b = pos[0], pos[1]
                    if a > 0:
                        if board[a-1][b] == state_type and [a-1, b] not in stone_group:
                            stone_group.append([a-1, b])

                    if a < 8:
                        if board[a+1][b] == state_type and [a+1, b] not in stone_group:
                            stone_group.append([a+1, b])

                    if b > 0:
                        if board[a][b-1] == state_type and [a, b-1] not in stone_group:
                            stone_group.append([a, b-1])

                    if b < 8:
                        if board[a][b+1] == state_type and [a, b+1] not in stone_group:
                            stone_group.append([a, b+1])

            return stone_group


cdef remove_group(group, white_score, black_score, board):
            if board[group[0][0]][group[0][1]] == 1:
                black_score = black_score + len(group)
            if board[group[0][0]][group[0][1]] == -1:
                white_score = white_score + len(group)
            for elmnt in group:
                board[elmnt[0]][elmnt[1]] = 0
            return board