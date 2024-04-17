import subprocess
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from model.model import Model  # Adjust your import path
import numpy as np
import gym
import copy

# Modify paths and import statements according to your project structure and requirements

def send_command(process, command):
    process.stdin.write(f"{command}\n")
    process.stdin.flush()

def get_response(process):
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("="):
            return line[2:].strip()

def init_gnugo(level=1, size=9, komi=7.5):
    command = ["gnugo", "--mode", "gtp", "--level", str(level), "--chinese-rules", "--komi", str(komi), "--boardsize", str(size)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    send_command(process, "boardsize " + str(size))
    send_command(process, "komi " + str(komi))
    return process

def coord_to_gtp_move(x, y):
    column = "ABCDEFGHJKLMNOPQRST"[x]
    row = y + 1
    return f"{column}{row}"

def play_move_gnugo(process, color, move):
    if move == 81:  # Pass
        send_command(process, f"play {color} pass")
    else:
        x, y = divmod(move, 9)
        gtp_move = coord_to_gtp_move(x, y)
        send_command(process, f"play {color} {gtp_move}")
    get_response(process)

def query_winner(process):
    send_command(process, "final_score")
    response = get_response(process)
    if response.startswith("B"):
        return 1  # Black wins
    elif response.startswith("W"):
        return -1  # White wins
    else:
        return 0  # Draw or error

def load_model(path):
    model = torch.load("/Users/greg/Repositories/teeny-go/pi-model-r12-c256-e1200.pt", map_location=torch.device('cpu'))
    model.eval()
    return model

def update_policy(model, optimizer, rewards, saved_log_probs):
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    for log_prob, R in zip(saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

def simulate_game(model, process):
    saved_log_probs = []
    rewards = []
    done = False
    env = gym.make('gym_go:go-v0', size=9, reward_method='heuristic')
    state = env.reset()
    while not done:
        state_tensor = torch.tensor(state[0:4], dtype=torch.float32).unsqueeze(0)
        logits, _ = model(state_tensor)
        logits = logits*torch.tensor(env.valid_moves()).float()
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        saved_log_probs.append(m.log_prob(action))
        next_state, _, done, _ = env.step(action.item())
        if not done:
            # Simulate GnuGo making a move
            play_move_gnugo(process, "white", action.item())
            _, _, done, _ = env.step(env.uniform_random_action())
        state = next_state
    winner = query_winner(process)
    if winner == 1:
        rewards = [1] * len(saved_log_probs)  # Model won
    else:
        rewards = [-1] * len(saved_log_probs)  # Model lost
    process.terminate()
    return saved_log_probs, rewards

def main():
    model_path = "pi-model-r12-c256-e1200.pt"
    model = load_model(model_path)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    n_games = 5
    for _ in tqdm(range(n_games)):
        process = init_gnugo(level=1)
        saved_log_probs, rewards = simulate_game(model, process)
        update_policy(model, optimizer, rewards, saved_log_probs)
    # Save your improved model
    # torch.save(model.state_dict(), "reinforced_model.pt")
    # print("Training Completed and Model Saved")

if __name__ == "__main__":
    main()