import torch

def get_action(model, state, env):

    # get current turn from env
    turn = env.turn()
    
    #  actions from policy
    action, value = model(state)

    # valid moves
    action = action*torch.tensor(env.valid_moves()).float()

    # set 81 to 0
    action[0][81] = 0

    # get the top 10 actions
    action = torch.topk(action, 20, dim=1).indices
    print(action)

    best_action = None
    best_dist = 2
    for i in range(len(action)):
        # make a copy of the env
        env_copy = copy.deepcopy(env)
        # take the action
        copy_state, _, _, _ = env_copy.step(action[0][i].item())
        # get the value of the new state
        _, new_value = model(torch.tensor(copy_state[0:4]).unsqueeze(0).float())
        new_value = new_value.item()
        # get the distance from the current value and turn
        distance = abs(new_value - turn)
        if distance < best_dist:
            best_dist = distance
            best_action = action[0][i].item()


    print('curr likely winner is ', value.item())

    return best_action


def get_k_actions(model, state, env, k=3):
    with torch.no_grad():
        action, _ = model(torch.tensor(state[0:4]).unsqueeze(0).float())
        action = action*torch.tensor(env.valid_moves()).float()
        action[0][81] = 0
        action = torch.softmax(action, dim=1)
        action = torch.topk(action, k, dim=1).indices
        action = action[0].tolist()
        print(action)
        return action


def explore_node(model, state, env, curr_depth=0, breadth=3, max_depth=3):
    with torch.no_grad():
        if curr_depth == max_depth:
            _, value = model(torch.tensor(state[0:4]).unsqueeze(0).float())
            return value.item()

        values = []
        # get top k actions do explore
        actions = get_k_actions(model, state, env, k=breadth)

        for a in actions:
            env_copy = copy.deepcopy(env)
            copy_state, _, _, _ = env_copy.step(a)
            values.append(explore_node(model, copy_state, env_copy, curr_depth+1, breadth=breadth, max_depth=max_depth))
        # get the current player
        curr_turn = env.turn()
        # return the value closest to the current player
        if curr_turn == 1:
            return max(values)
        return min(values)
    

def mcts(model, state, env, max_depth=3, breadth=3):
    '''monte carlo tree search'''
    with torch.no_grad():        
        actions = get_k_actions(model, state, env, k=breadth)

        best_move = None
        min_value = 2
        curr_player = env.turn()

        for a in actions:
            env_copy = copy.deepcopy(env)
            copy_state, _, _, _ = env_copy.step(a)
            value = explore_node(model, copy_state, env_copy, curr_depth=0, breadth=breadth, max_depth=max_depth)
            distance = abs(value - curr_player)
            if distance < min_value:
                min_value = distance
                best_move = a
        return best_move
    

