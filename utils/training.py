import torch
from tqdm import tqdm
import os

def load_dataset(path, start_idx, end_idx):
    """
    Load dataset from specified path, starting from start_idx and ending at end_idx.

    Args:
        path (str): The path to the dataset.
        start_idx (int): The starting index of the dataset.
        end_idx (int): The ending index of the dataset.

    Returns:
        tuple: A tuple containing the loaded dataset, consisting of x, y, and reward.
    """
    if not os.path.exists(path):
        raise Exception(f"Path {path} does not exist")

    x = []
    y = []
    reward = []

    x_path = path + "state-"
    y_path = path + "action-"
    r_path = path + "reward-"

    print('loading games:')
    for i in tqdm(range(start_idx, end_idx)):
        try:
            x.append(torch.load(x_path+str(i)+".pt"))
            y.append(torch.load(y_path+str(i)+".pt"))
            r.append(torch.load(r_path+str(i)+".pt"))
            
        except:
            pass
    
    # binary data in int8 to save memory
    x = torch.cat(x).to(dtype=torch.int8)
    y = torch.cat(y).to(dtype=torch.int8)
    r = torch.cat(r).to(dtype=torch.int8).unsqueeze(1)

    return x, y, reward


def policy_loss(logits, one_hot_targets):
    """
    Calculates the policy loss given the logits and one-hot targets.

    Args:
        logits (torch.Tensor): The predicted logits.
        one_hot_targets (torch.Tensor): The one-hot encoded targets.

    Returns:
        torch.Tensor: The calculated policy loss.
    """
    log_probs = torch.log_softmax(logits, dim=1) * one_hot_targets        
    loss = -torch.sum(log_probs, dim=1)
    loss = torch.mean(loss)
    return loss


def joint_loss(logits, value, y, r, alpha=0.01):
    """
    Calculates the joint loss for policy and value predictions.

    Args:
        logits (Tensor): The predicted policy logits.
        value (Tensor): The predicted value.
        y (Tensor): The target policy.
        r (Tensor): The target value.
        alpha (float, optional): The weight for the value loss. Defaults to 0.01.

    Returns:
        Tensor: The joint loss.
        Tensor: The policy loss.
        Tensor: The value loss.
    """

    policy_loss = policy_loss(logits, y)
    value_loss = torch.nn.MSELoss(value, r)
    return policy_loss + alpha * value_loss, policy_loss, value_loss


def transform_board_and_action(state, action, num_rotation, num_flips):
    """
    Transforms the board state and action based on the specified number of rotations and flips.

    Args:
        state (torch.Tensor): The board state.
        action (torch.Tensor): The action.
        num_rotation (int): The number of rotations to apply.
        num_flips (int): The number of flips to apply.

    Returns:
        torch.Tensor: The transformed board state.
        torch.Tensor: The transformed action.
    """
    pass_move = action[:, -1].unsqueeze(-1)  
    action = action[:, :-1] 

    for _ in range(num_rotation):
        state = state.rot90(1, [-2, -1])  
        action = action.reshape(-1, 9, 9).rot90(1, [-2, -1]).reshape(-1, 81)

    if num_flips == 1:
        state = state.flip(-1) 
        action = action.reshape(-1, 9, 9).flip(-1).reshape(-1, 81)

    action = torch.cat((action, pass_move), dim=1)

    return state, action


def shuffle_dataset(x, y, r):
    """
    Shuffles the dataset.

    Args:
        x (torch.Tensor): The input data.
        y (torch.Tensor): The target data.
        r (torch.Tensor): The reward data.

    Returns:
        tuple: A tuple containing the shuffled dataset.
    """
    rand_perm = torch.randperm(x.shape[0])
    x = x[rand_perm]
    y = y[rand_perm]
    r = r[rand_perm]

    return x, y, r


def print_model_params(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(f"num params: {num_params}")  