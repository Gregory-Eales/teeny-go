import numpy as np

class RandomAgent(object):

    def __init__(self):

        self.elo_rating = 0

    def make_move(self, valid_moves):
        return np.random.choice(list(range(82)),p=valid_moves)
        
        
class MultiRandomAgent(object):

    def __init__(self):
        
        self.elo_rating = 0

    def make_move(self, valid_moves):
        return np.random.choice(list(range(82)),p=valid_moves)

    def make_moves(self, valid_moves):
        moves = []
        for move in range(valid_moves):
            moves.append(self.make_move(move))

        return moves
        
    
