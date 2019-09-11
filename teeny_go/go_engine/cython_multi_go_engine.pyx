from cython_go_engine import GoEngine
import numpy as np
cimport numpy as np
import time

cdef class MultiGoEngine():
    cpdef int num_games
    cpdef list active_games
    cpdef dict games
    cpdef public np.ndarray move_tensor

    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.move_tensor = np.zeros([1,1])
        self.generate_game_objects()

##########################
# Main Game Step Methods #
##########################

    cpdef np.ndarray take_game_step(self, move_tensor):

        self.input_move_tensor(move_tensor)

        self.remove_invalid_moves()

        self.make_moves()

        self.remove_inactive_games()

        return self.get_game_states()

    cpdef input_move_tensor(self, move_tensor):
        self.move_tensor = move_tensor

    cpdef remove_invalid_moves(self):

        cdef list invalid_move_tensor = []
        cdef str game
        for game in self.active_games:
            invalid_move_tensor.append(self.games[game].get_invalid_moves())
        self.move_tensor[:,0:81] = self.move_tensor[:,0:81] - np.concatenate(invalid_move_tensor)

    cpdef make_moves(self):

        # choose move based on weighted probability
        # check to see if move is pass or not
        # if pass, pass
        # else: make move

        cdef int num
        cdef str game
        cdef list moves
        cdef int move

        for num, game in enumerate(self.active_games):

            moves = list(range(82))
            move = np.random.choice(moves, p=self.move_tensor[num][0:82]/np.sum(self.move_tensor[num][0:82]))

            if move == 81:
                self.games[game].make_pass_move()

            else:
                self.games[game].make_move([move//9, move%9])


    cpdef get_active_game_states(self):
        cdef list states_tensor = []
        cdef str game
        for game in self.active_games:
            states_tensor.append(self.games[game].get_board_tensor())
        return np.concatenate(states_tensor)

    cpdef np.ndarray get_all_game_states(self):

        cdef list states_tensor = []
        cdef int i

        for i in range(self.num_games):
            states_tensor.append(self.games["G"+str(i)].get_board_tensor())
        return np.concatenate(states_tensor)

    cpdef remove_inactive_games(self):

        cdef str game

        for game in self.active_games:
            if self.games[game].is_playing == False:
                self.active_games.remove(game)


    #######################
    # Misc Engine Methods #
    #######################

    def generate_game_objects(self):

        cdef int i

        for i in range(self.num_games):
            self.games["G"+str(i)] = GoEngine()
            self.active_games.append("G"+str(i))

    def reset_games(self):

        cdef int i

        for i in range(self.num_games):
            self.games["G"+str(i)].new_game()



def main():
    n = 5000
    mge = MultiGoEngine(num_games=n)
    mge.move_tensor = np.ones([n, 83])
    t = time.time()
    mge.get_active_game_states()
    mge.remove_invalid_moves()
    mge.make_moves()
    mge.remove_inactive_games()
    print("Game Step Time:", round(time.time()-t, 3), "s")

if __name__ == "__main__":
    main()
