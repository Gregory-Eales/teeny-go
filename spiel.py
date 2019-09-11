from teeny_go.go_engine.open_spiel.build_python_3.python import pyspiel as ps

import time





board_size = {"board_size": ps.GameParameter(9)}
game = ps.load_game("go", board_size)

state = game.new_initial_state()

print(state.information_state())

is_playing = True
t = time.time()



for i in range(150):


    legal_actions = state.legal_actions()
    #print(legal_actions)
    if len(legal_actions) > 4:
        state.apply_action(legal_actions[-1])
    else:
        is_playing = False


#print(state.history())

print(1/(time.time()-t), "games/second")
