import numpy as np
import time
import pyspiel

class MultiGoEngine(object):

    def __init__(self, num_games=100):
        self.num_games = num_games
        self.active_games = []
        self.games = {}
        self.move_tensor = None
        self.generate_game_objects()

    def train(self, agent, num_games, train_time):

        # get start of training time
        start_time = time.time()

        # main playthrough loop
        while ((time.time()-start_time)/(60*60) < train_time):

            while self.is_playing_game():
                move_tensor = agent.forward(self.get_active_game_states)
                self.take_game_step(move_tensor)

            self.save_game_data()

            agent.optimize(self.x, self.y, iterations)

            self.clear_game_cache()

    def save_game_data(self):
        pass

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
            states_tensor.append(self.games[game].get_board_tensor())
        return np.concatenate(states_tensor)

    def get_all_game_states(self):
        states_tensor = []
        for i in range(self.num_games):
            states_tensor.append(self.games["G"+str(i)].get_board_tensor())
        return np.concatenate(states_tensor)

    def remove_inactive_games(self):

        for game in self.active_games:
            if self.games[game].is_playing == False:
                self.active_games.remove(game)

    def generate_game_objects(self):

        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = ps.load_game("go", board_size)

        for i in range(self.num_games):
            self.games["G"+str(i)] = game.new_initial_state()
            self.active_games.append("G"+str(i))

    def reset_games(self):
        for i in range(self.num_games):
            self.games["G"+str(i)].new_game()

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
