import numpy as np

from disco.disco import Engine
from disco import gtp, go



class DiscoAgent(object):


    def __init__(self):
        
        self.disco = Engine()



    def convert_game_tensor(self, state):

        EMPTY, WHITE, BLACK = 0, 1, 2

        board = go.Board()

        if state[2][0][0] == 0:
            board.color = BLACK
            turn = BLACK
            b_idx = 0
            w_idx = 1

        else:
            board.color = WHITE
            turn = WHITE
            b_idx = 1
            w_idx = 0

        for idx, square in enumerate(board.squares):

            x, y = divmod(square.pos, 9)

            if state[b_idx][y][x] != 0:
                #board.squares[idx].color = BLACK
                board.squares[idx].move(BLACK)

            if state[w_idx][y][x] != 0:
                #board.squares[idx].color = WHITE
                board.squares[idx].move(WHITE)

        return board, turn
        

    def act(self, state):


        board, turn = self.convert_game_tensor(state)

        self.disco.board = board

        action = self.disco.genmove(turn)

        return action

def main():

    da = DiscoAgent()

    import gym
    import argparse

    parser = argparse.ArgumentParser(description='Demo Go Environment')
    parser.add_argument('--randai', action='store_true')
    parser.add_argument('--boardsize', type=int, default=9)
    args = parser.parse_args()

    go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')
    done = False
    state = go_env.reset()
    state = state[0:3].reshape([3, 9, 9])

    for i in range(100):

        try:
            action = da.act(state)
            state, reward, done, _ = go_env.step(action)
            s = state[0:3].reshape([3, 9, 9])
                
                
        except Exception as e:
            print(e)
            continue
            

        if True:

            if go_env.game_ended():
                break
            #action = go_env.uniform_random_action()
            action = go_env.render("human")
            state, reward, done, _ = go_env.step(action)
            state = state[0:3].reshape([3, 9, 9])

    

if __name__ == "__main__":
    main()