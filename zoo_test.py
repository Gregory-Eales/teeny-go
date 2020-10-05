from pettingzoo.classic import go_v0
import random
import time

from pettingzoo.classic import texas_holdem_no_limit_v0


env = texas_holdem_no_limit_v0.env(player)


print(env.reset().shape)
"""
env = go_v0.env(board_size = 9, komi = 7.5)



def move(info, agent):
	move = random.choice(env.infos[agent]['legal_moves'])
	if move >= 82:
		move = 81
	return move

print(env.reset().shape)



counter = 0
t = time.time()
for i in range(100):
	observation = env.reset()
	for agent in env.agent_iter():
		counter+=1
		try:
		    reward, done, info = env.last()
		    observation = env.step(move(info, agent))

		except:
			break
t = time.time()-t
print(t)
print(1/(50*t/counter))
print(counter)
"""
