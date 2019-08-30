from go_engine import GoEngine
import numpy as np
import time

class MultiGoEngine(object):


    # 1. get move tensor -> (n, 83)
    # 2. get tensor of invalid moves from game engine
    # 3. take away invalid moves from move tensors
    # 4. pick move based on weighted probabilities
    # 5. make moves
    # 6. remove inactive games
    # 6. get board state tensor
    # 7. return state tensor

    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.move_tensor = None
        self.generate_game_objects()

        ##########################
        # Main Game Step Methods #
        ##########################

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
            self.move_tensor - np.concatenate(invalid_move_tensor)

    def make_moves(self):
        pass

    def get_active_game_states(self):
        states_tensor = []
        for game in self.active_games:
            states_tensor.append(self.games[game].get_board_tensor())
        return np.concatenate(states_tensor)

    def get_all_game_states(self):
        states_tensor = []
        for i in range(self.num_games):
            states_tensor.append(self.games["G"+str(i)].get_board_tensor())
        return np.concatenate(states_tensor)



        #######################
        # Misc Engine Methods #
        #######################

    def generate_game_objects(self):
        for i in range(self.num_games):
            self.games["G"+str(i)] = GoEngine()
            self.active_games.append("G"+str(i))

    def reset_games(self):
        for i in range(self.num_games):
            self.games["G"+str(i)].new_game()

def main():
    mge = MultiGoEngine(num_games=120*1000)
    mge.get_active_game_states()
    #time.sleep(30)

if __name__ == "__main__":
    main()
