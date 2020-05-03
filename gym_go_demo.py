import gym
import argparse
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='Demo Go Environment')
parser.add_argument('--randai', action='store_true')
parser.add_argument('--boardsize', type=int, default=9)
args = parser.parse_args()

go_env = gym.make('gym_go:go-v0',size=args.boardsize,reward_method='heuristic')
done = False

rewards = []

while not done:
    #action = go_env.render('terminal')
    action = go_env.uniform_random_action()
    try:
        state, reward, done, _ = go_env.step(action)
    except Exception as e:
        print(e)
        continue
    if args.randai:
        if go_env.game_ended():
            break
        action = go_env.uniform_random_action()
        state, reward, done, _ = go_env.step(action)

    rewards.append(reward)
#go_env.render(mode="terminal")

print(state.shape)

plt.plot(rewards)
plt.show()