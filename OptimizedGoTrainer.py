import numpy as np
from TeenyGoAI import NeuralNetwork

class GoTrainer(object):

    def __init__(self):
        self.board = np.zeros([9, 9])
        self.move_holder = np.zeros([9, 9])
        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.white_score = 0
        self.black_score = 0
        self.y_data = []
        self.x_data = []
        self.turn = "black"
        size = np.zeros([9, 9])
        self.NN = NeuralNetwork(size, size, alpha=0.001, iterations=1, num_layers=3, hidden_addition=1, lamb=0.0000)

    # returns an empty board with the move about to be made
    def position_to_coordinates(self, move):
        #self.move_holder = np.zeros([9,9])
        #self.move_holder[self.letters.index(move[0]), int(move[1])]
        return [self.letters.index(move[0]), int(move[1])]

    def get_y(self, move):
        y = self.board * 1
        y[y==0] = 0.5
        y[y!=0.5] = 0
        y[move[0]][move[1]] = 1
        return y

    def play(self, data, winner):
        self.board = np.zeros([9, 9])
        self.white_score = 0
        self.black_score = 0
        for move in data:
            type_for_capture = 0
            move = self.position_to_coordinates(move)
            x = self.board*1
            y = self.get_y(move)
            if self.board[move[0]][move[1]] == 0:
                # stand off is set to False
                #print("made it")
                stand_off = 0
                #if the position is good then...
                if move is not None:
                    move_state = self.board[move[0]][move[1]]
                    # if the board space is empty...
                    if move_state == 0:
                        # place white or black depending on the turn
                        if self.turn == "white":
                            self.board[move[0]][move[1]] = 1
                        elif self.turn == "black":
                            self.board[move[0]][move[1]] = -1
                        # if
                        check_captures = self.check_capture_pieces(move)

                        if check_captures == "white" or check_captures == "black":
                            if check_captures == "white":
                                type_for_capture = 1
                                stand_off = 1

                            if check_captures == "black":
                                type_for_capture = -1
                                stand_off = 1

                        if check_captures == 0 or stand_off == 1:
                            if self.turn == "white":
                                self.turn = "black"
                                #print("its blacks turn")
                            elif self.turn == "black":
                                self.turn = "white"
                                #print("its whites turn")

                        if True:
                            #print("optimizing")
                            #self.NN.optimize()
                            if self.turn == winner:
                                self.x_data.append(x)
                                self.y_data.append(y)

                        if check_captures == 1:
                            #print(move)
                            #print("Invalid Move")
                            #print(self.board)
                            #print(move)
                            self.board[move[0]][move[1]] = 0
                            #print(self.board)

            if type_for_capture != 0:
                self.capture_pieces(type_for_capture)
        #print(self.board)

    def capture_pieces(self, type_for_capture):
        for i in range(9):
            for j in range(9):
                location_state = self.board[i][j]
                if location_state != 0 and location_state == type_for_capture:
                    group = self.get_group([i, j], location_state)
                    if group != []:
                        free = self.check_neighbors(group, location_state)
                        if free == "False":
                            self.remove_group(group)

    def check_capture_pieces(self, position):
        killing_itself = 0
        for i in range(9):
            for j in range(9):
                location_state = self.board[i, j]
                if location_state != 0:
                    group = self.get_group([i, j], location_state)
                    if group != []:
                        free = self.check_neighbors(group, location_state)
                        if free == "False":
                                if position in group:
                                    killing_itself = 1
                                if location_state == 1 and self.board[position[0]][position[1]] != 1:
                                    return "white"
                                if location_state == -1 and self.board[position[0]][position[1]] != -1:
                                    return "black"

        return killing_itself


    def check_neighbors(self, group, state_type):
        liberty = "False"

        for position in group:

            a, b = position[0], position[1]

            if a < 8:
                if self.board[a+1][b] == 0:
                    return True

            if a > 0:
                if self.board[a-1][b] == 0:
                    return True

            if b < 8:
                if self.board[a][b+1] == 0:
                    return True

            if b > 0:
                if self.board[a][b-1] == 0:
                    return True

        return liberty


    def get_group(self, position, state_type):
        stone_group = []
        stone_group.append(position)
        for j in range(20):
            for pos in stone_group:
                a, b = pos[0], pos[1]
                if a > 0:
                    if self.board[a-1][b] == state_type and [a-1, b] not in stone_group:
                        stone_group.append([a-1, b])

                if a < 8:
                    if self.board[a+1][b] == state_type and [a+1, b] not in stone_group:
                        stone_group.append([a+1, b])

                if b > 0:
                    if self.board[a][b-1] == state_type and [a, b-1] not in stone_group:
                        stone_group.append([a, b-1])

                if b < 8:
                    if self.board[a][b+1] == state_type and [a, b+1] not in stone_group:
                        stone_group.append([a, b+1])

        return stone_group

    def remove_group(self, group):
        #print(group)
        if self.board[group[0][0]][group[0][1]] == 1:
            self.black_score = self.black_score + len(group)
        if self.board[group[0][0]][group[0][1]] == -1:
            self.white_score = self.white_score + len(group)
        #print("White Score: " + str(self.white_score))
        #print("Black Score: " + str(self.black_score))
        for elmnt in group:
            self.board[elmnt[0]][elmnt[1]] = 0
