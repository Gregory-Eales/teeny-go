import numpy as np
import time
import pyspiel

class MultiGoEngine(object):

    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.game_states = {}
        self.game_x_data = {}
        self.game_y_data = {}
        self.move_tensor = None
        self.generate_game_objects()
        self.move_map = self.get_move_map()

    def get_move_map(self):
        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = pyspiel.load_game("go", board_size)

        state = game.new_initial_state()
        return state.legal_actions()


    def is_playing_games(self):
        return len(self.active_games)>0

    def finalize_game_data(self):

        for game in self.games.keys():

            self.game_x_data[game] = np.concatenate(self.game_x_data[game])
            self.game_y_data[game] = np.concatenate(self.game_y_data[game])

            rewards = self.games[game].returns()

            # if black wins
            if rewards[0] > rewards[1]:
                self.game_y_data["game"]

            # if white wins
            if rewards[0] < rewards[1]:
                pass

            # if draw
            else:
                pass

    def take_game_step(self, move_tensor):

        # internalize move_tensor
        self.input_move_tensor(move_tensor)

        # removes invalid moves
        self.remove_invalid_moves()

        # makes moves
        self.make_moves()

        # remove terminal games from active games
        self.remove_inactive_games()

    def input_move_tensor(self, move_tensor):
        self.move_tensor = move_tensor

    def remove_invalid_moves(self):

        valid_move_tensor = []

        for game in self.active_games:
            valid_moves = self.games[game].legal_actions_mask()
            valid_moves = np.array(valid_moves[0:441]).reshape(21, 21)
            valid_moves = valid_moves[1:10,1:10].reshape(81)
            valid_moves = np.append(valid_moves, 1)
            valid_move_tensor.append(valid_moves)

        valid_move_tensor = np.array(valid_move_tensor)
        self.move_tensor[:,0:82] = self.move_tensor[:,0:82] * valid_move_tensor

    def make_moves(self):

        for num, game in enumerate(self.active_games):
            self.game_y_data[game].append(self.move_tensor[num])
            moves = list(range(82))
            move = np.random.choice(moves, p=self.move_tensor[num][0:82]/np.sum(self.move_tensor[num][0:82]))
            self.games[game].apply_action(self.move_map[int(move)])



    def get_active_game_states(self):

        states_tensor = []
        for game in self.active_games:
            state = self.games[game].observation_as_normalized_vector()
            state = np.array(state).reshape(-1, 81)
            state = (state[0] + state[1]*-1)
            self.game_states[game].append(state.reshape(1, 9, 9))
            state_tensor = self.generate_state_tensor(game)
            self.game_x_data[game].append(state_tensor)
            states_tensor.append(state_tensor)
        return np.concatenate(states_tensor)

    def generate_state_tensor(self, game):

        black = []
        white = []
        turn = self.games[game].current_player()

        if turn == 1:
            turn = [np.zeros([1, 9, 9])]

        elif turn == 0:
            turn = [np.ones([1, 9, 9])]

        else:
            print(turn)

        for i in range(1, 6):
            black.append(np.where(self.game_states[game][-i] == 1, 1, 0).reshape(1, 9, 9))
            white.append(np.where(self.game_states[game][-i] == -1, 1, 0).reshape(1, 9, 9))

        black = np.concatenate(black, axis=0)

        white = np.concatenate(white, axis=0)
        turn = np.concatenate(turn, axis=0)

        output = np.concatenate([black, white, turn]).reshape(1, 11, 9, 9)

        return output

    def get_all_game_states(self):
        states_tensor = []
        for i in range(self.num_games):
            states_tensor.append(self.games["G"+str(i)].get_board_tensor())
        return np.concatenate(states_tensor)

    def remove_inactive_games(self):

        for game in self.active_games:
            if self.games[game].is_terminal() == True:
                self.active_games.remove(game)

    def generate_game_objects(self):

        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = pyspiel.load_game("go", board_size)

        for i in range(self.num_games):
            self.games["G"+str(i)] = game.new_initial_state()
            self.game_states["G"+str(i)] = []
            self.game_x_data["G"+str(i)] = []
            self.game_y_data["G"+str(i)] = []
            for j in range(7):
                self.game_states["G"+str(i)].append(np.zeros([9,9]))
            self.active_games.append("G"+str(i))

    def reset_games(self, num_games=None):
        if self.num_games == None: pass
        else: self.num_games = num_games
        del(self.games)
        del(self.game_states)
        del(self.game_x_data)
        del(self.game_y_data)
        self.generate_game_objects()

def main():
    n = 1000
    mge = MultiGoEngine(num_games=n)
    mge.move_tensor = np.ones([n, 82])
    t = time.time()
    for i in range(50):
        ag = len(mge.active_games)
        if ag > 0:
            mge.move_tensor = np.ones([ag, 82])
            mge.get_active_game_states()
            mge.remove_invalid_moves()
            mge.make_moves()
            mge.remove_inactive_games()
    print("Game Step Time:", round(time.time()-t, 3), "s")


if __name__ == "__main__":
    main()
