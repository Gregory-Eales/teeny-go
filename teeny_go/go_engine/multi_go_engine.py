from go_engine import GoEngine
import numpy as np

class MultiGoEngine(object):

    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.move_tensor = None

        self.generate_game_objects()

    def generate_game_objects(self):
        for i in range(self.num_games):
            self.games["G"+str(i)] = GoEngine()
            self.active_games.append("G"+str(i))

    def reset_games(self):
        for i in range(self.num_games):
            self.games["G"+str(i)].new_game()

    def input_move_tensor(self, move_tensor):
        self.move_tensor = move_tensor

    def get_game_states(self):
        states_tensor = []
        for i in range(self.num_games):
            states_tensor.append(self.games["G"+str(i)].get_board_tensor())
        return np.concatenate(states_tensor)

    def take_game_step(self):

        for game in self.active_games:
            self.games[game].check_valid()


def main():
    mge = MultiGoEngine()
    mge.get_game_states()

if __name__ == "__main__":
    main()
