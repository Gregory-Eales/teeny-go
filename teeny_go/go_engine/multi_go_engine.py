import numpy as np
import time
import pyspiel

class MultiGoEngine(object):

    # initialize games
    # get game states from active game objects
    # translate game states into input tensors [1, 11, 9, 9]
    # save input tensors
    # make predictions using neural net
    # use prob selection to make moves
    # check to see if game is terminal
    # if game terminal remove from active games


    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.game_states = {}
        self.game_data = {}
        self.move_tensor = None
        self.generate_game_objects()

    def is_playing_games(self):
        return len(self.active_games)>0

    def take_game_step(self, move_tensor):

        self.input_move_tensor(move_tensor)

        self.remove_invalid_moves()

        self.make_moves()

        self.remove_inactive_games()

        return self.get_game_states()

    def input_move_tensor(self, move_tensor):
        self.move_tensor = move_tensor

    def remove_invalid_moves(self):
        invalid_move_tensor = []
        for game in self.active_games:
            invalid_move_tensor.append(self.games[game].get_invalid_moves())
        self.move_tensor[:,0:81] = self.move_tensor[:,0:81] - np.concatenate(invalid_move_tensor)

    def make_moves(self):

        for num, game in enumerate(self.active_games):

            moves = list(range(82))
            move = np.random.choice(moves, p=self.move_tensor[num][0:82]/np.sum(self.move_tensor[num][0:82]))

            if move == 81:
                self.games[game].make_move(move)

            else:
                self.games[game].make_move(move)


    def get_active_game_states(self):
        states_tensor = []
        for game in self.active_games:
            state = self.games[game].information_state_as_normalized_vector()
            states_tensor.append(state)
            self.game_states[game].append(state)
        return np.concatenate(states_tensor)

    def get_all_game_states(self):
        states_tensor = []
        for i in range(self.num_games):
            states_tensor.append(self.games["G"+str(i)].get_board_tensor())
        return np.concatenate(states_tensor)

    def remove_inactive_games(self):

        for game in self.active_games:
            if self.games[game].is_terminal == True:
                self.active_games.remove(game)

    def generate_game_objects(self):

        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = ps.load_game("go", board_size)

        for i in range(self.num_games):
            self.games["G"+str(i)] = game.new_initial_state()
            self.game_states["G"+str(i)] = []
            self.game_data["G"+str(i)] = []
            for i in range(7):
                self.game_data["G"+str(i)].append(np.zeros([9,9]))
            self.active_games.append("G"+str(i))

    def reset_games(self, num_games=self.num_games):
        self.num_games = num_games
        del(self.games)
        del(self.game_states)
        del(self.game_data)
        self.generate_game_objects()

def main():
    n = 2000
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
