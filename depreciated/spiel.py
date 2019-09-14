import pyspiel as ps
import numpy as np
import time





board_size = {"board_size": ps.GameParameter(9)}
game = ps.load_game("go", board_size)

state1 = game.new_initial_state()
state2 = game.new_initial_state()

print(state1.legal_actions())
x = state1.legal_actions_mask()

x = np.array(x[0:441]).reshape(1, 21, 21)
print(x[:,1:10,1:10].shape)

is_playing = True
t = time.time()

board = []

for i in range(9):
    board.append(np.zeros([9,9]))

for i in range(150):

    black = []
    white = []
    turn = None
    print(state1.current_player())
    if turn == "white":
        turn = [np.zeros([9, 9])]
    else:
        turn = [np.ones([9, 9])]

    for i in range(1, 6):
        black.append(np.where(board[-i] == 1, 1, 0))
        white.append(np.where(board[-i] == -1, 1, 0))
    a = np.array(black+white+turn)

    legal_actions = state1.legal_actions()

    print("legals actions:")
    print(len(legal_actions))
    #print(legal_actions)
    if len(legal_actions) > 4:
        state1.apply_action(legal_actions[-1])
    else:
        is_playing = False

    board.append(np.array(state1.information_state_as_normalized_vector()))





print(1/60*(time.time()-t), "games/minute")
