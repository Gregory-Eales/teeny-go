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
        invalid_move_tensor = []
        for game in self.active_games:
            invalid_move_tensor.append(self.games[game].get_legal_actions())
        self.move_tensor[:,0:81] = self.move_tensor[:,0:81] * np.concatenate(invalid_move_tensor)

    def make_moves(self):

        for num, game in enumerate(self.active_games):
            self.game_y_data[game].append(self.move_tensor[num])
            moves = list(range(82))
            move = np.random.choice(moves, p=self.move_tensor[num][0:82]/np.sum(self.move_tensor[num][0:82]))
            self.games[game].apply_action(move)


    def get_active_game_states(self):

        states_tensor = []
        for game in self.active_games:
            state = self.games[game].information_state_as_normalized_vector()
            state_tensor = self.generate_state_tensor(game, state)
            self.game_x_data[game].append(state_tensor)
            states_tensor.append(state_tensor)
            self.game_states[game].append(state)
        return np.concatenate(states_tensor)

    def generate_state_tensor(self, game, state):

        black = []
        white = []
        turn = None
        turn = self.games[game].current_player()

        if turn == 1:
            turn = [np.zeros([9, 9])]

        else:
            turn = [np.ones([9, 9])]

        for i in range(1, 6):
            black.append(np.where(self.game_states[game][-i] == 1, 1, 0))
            white.append(np.where(self.game_states[game][-i] == -1, 1, 0))

        return np.array(black+white+turn).reshape([1, 11, 9, 9])

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
        del(self.game_x_data)
        del(self.game_y_data)
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
