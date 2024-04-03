import torch
import gym
import numpy as np
import sys
import argparse
# import deep copy
import copy

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


# need to add the path to the sys path
sys.path.append("/Users/greg/Repositories/teeny-go")


from model.model import Model

def load_model():
    model = torch.load("/Users/greg/Repositories/teeny-go/pi-model-r12-c256-e100.pt", map_location=torch.device('cpu'))
    print("loaded model")
    return model


def get_action(policy, critic, state, env):

    # get current turn from env
    turn = env.turn()
    
    #  actions from policy
    action, _ = policy(state)
    _, value = critic(state)

    # valid moves
    action = action*torch.tensor(env.valid_moves()).float()

    # set 81 to 0
    action[0][81] = 0

    # get the top 10 actions
    action = torch.topk(action, 10, dim=1).indices
    print(action)

    best_action = None
    best_dist = 2
    for i in range(10):
        # make a copy of the env
        env_copy = copy.deepcopy(env)
        # take the action
        copy_state, _, _, _ = env_copy.step(action[0][i].item())
        # get the value of the new state
        _, new_value = critic(torch.tensor(copy_state).unsqueeze(0).float())
        new_value = new_value.item()
        # get the distance from the current value and turn
        distance = abs(new_value - turn)
        if distance < best_dist:
            best_dist = distance
            best_action = action[0][i].item()


    print('curr likely winner is ', best_dist)

    return best_action


def gtp():
    model = load_model()

    #value_model = torch.load("/Users/greg/Repositories/teeny-go/value-model-r12-c256-e50.pt", map_location=torch.device('cpu'))

    env = gym.make('gym_go:go-v0',size=9,reward_method='real')
    print("loaded env and model...")

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "j", "t"]
    letter_to_index = {letters[i].upper(): i for i in range(len(letters))}

    state = env.reset()
    has_passed = False

    while True:
        
        # wait for text input
        command = input().strip().split()

        if command[0] == "quit":
            print('= \n\n')
            break

        elif command[0] == "protocol_version":
            print('=%s %s\n\n' % ('', 2,), end='')

        elif command[0] == "version":
            print('=%s %s\n\n' % ('', 1.0,), end='')

        elif command[0] == "name":
            print('=%s %s\n\n' % ('', "teeny-go",), end='')

        elif command[0] == "clear_board":
            print('= \n\n')

        # play a move
        elif command[0] == "play":
            if command[2].lower() == 'pass':
                env.step(81)
                has_passed = True
            else:
                env.step(letter_to_index[command[2][0]] + 9 * (9 - int(command[2][1])))
                env.render(mode="terminal")
            print('= \n\n')

            # move = letter_to_index[move[0]] + 9 * (9 - int(move[1]))

        # genmove
        elif command[0] == "genmove":

            if has_passed:
                move = env.step(81)
                has_passed = False
                # responde with a0
                print('=%s %s\n\n' % ('', 'a0',), end='')
            else:
                with torch.no_grad():
                    state = torch.tensor([state[0:4]]).float()
                    #print(state.shape)
                    action, _ = model(state)

                    # apply softmax to action
                    action = torch.softmax(action, dim=1)

                    action = action*torch.tensor(env.valid_moves()).float()
                    #action[0][81] = 0
                    
                    # top action, 
                    action = torch.argmax(action, dim=1).item()

                    #action = get_action(model, value_model, state, env)

                    # # # randomly select top 3 actions, first get top 3 actions
                    # # action = torch.topk(action, 3, dim=1).indices
                    # # # then randomly select one of the top 3
                    # # action = action[0][np.random.randint(3)].item()


                    state, reward, done, info = env.step(action)
                    print('=%s %s\n\n' % ('', letters[action % 9] + str(9 - action // 9),), end='')
                    print(action)
                    print(state[0] + state[1])
                    print('--------------------')
                    # remainder + 

            #print('=%s %s\n\n' % ('', letters[action % 9] + str(9 - action // 9),), end='')

        elif command[0] == "clear_board":
            state = env.reset()
            has_passed = False
            print('= \n\n')

        else:
            print("unknown command")
            print("= \n\n")
        


    print("quitting gtp loop")







def main():
    model = load_model()
    # print num params
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"num params: {num_params}")


if __name__ == "__main__":
    gtp()