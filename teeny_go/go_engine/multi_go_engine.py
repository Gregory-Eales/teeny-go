from go_engine import GoEngine
import numpy as np

class MultiGoEngine(object):

    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.move_tensor = None

    def generate_game_objects(self):
        for i in range(self.num_games):
            self.games["G"+str(i)] = GoEngine()
            self.active_games.append("G"+str(i))

    def reset_games(self):
        for i in range(self.num_games):
            self.games["G"+str(i)].new_game()

    def input_move_tensor(self, move_tensor):
        self.move_tensor = move_tensor

    def take_game_step(self):
        pass

    def get_game_states(self):
        pass
