from matplotlib import pyplot as plt
from os import listdir
from tqdm import tqdm

def populate_ranks():

	ranks_dist = {}

	ranks_dist["?"] = 0

	for i in reversed(range(1, 31)):

		ranks_dist["{}k".format(i)] = 0

	for i in range(1, 10):

		ranks_dist["{}d".format(i)] = 0

	return ranks_dist

def get_player_ranks(path="/"):

	file = open(path, encoding="utf8", errors='ignore')
	lines = [line.strip() for line in file.readlines()]

	white_rank = None
	black_rank = None

	for line in lines:

		if line[0:2] == "BR":
			black_rank = line[3:-1]

		if line[0:2] == "WR":
			white_rank = line[3:-1]

		if white_rank != None and black_rank != None:
			break

	if white_rank == None and black_rank == None:
			pass

	return white_rank, black_rank


def get_sgf_rank_dist(path="./data/ogs_games/"):


	file_paths = listdir(path)

	# rank ditribution
	white_dist = populate_ranks()
	black_dist = populate_ranks()
	total_dist = populate_ranks()


	for game_path in tqdm(file_paths, "getting rank dist: "):
		white_rank, black_rank = get_player_ranks(path=path+game_path)

		if white_rank != None and black_rank != None:

			try:
				white_dist[white_rank] += 1
				total_dist[white_rank] += 1

			except:
				white_dist['?'] += 1
				total_dist['?'] += 1

			try:
				black_dist[black_rank] += 1
				total_dist[black_rank] += 1

			except:	
				black_dist['?'] += 1
				total_dist['?'] += 1


	return total_dist, white_dist, black_dist


def plot_rank_dist(path="./data/ogs_games/"):

	total_dist, white_dist, black_dist = get_sgf_rank_dist(path=path)

	keys = total_dist.keys()
	values = total_dist.values()

	plt.bar(keys, values)
	plt.ylabel('Players Per Rank')
	plt.xlabel('Go Ranks')
	plt.xticks(fontsize=6)
	plt.title('Go Player Rank Distribution')
	plt.show()

	black_keys = black_dist.keys()
	black_values = black_dist.values()

	plt.bar(black_keys, black_values)

	white_keys = white_dist.keys()
	white_values = white_dist.values()

	plt.bar(white_keys, white_values)

	plt.ylabel('Players Per Rank')
	plt.xlabel('Go Ranks')
	plt.xticks(fontsize=6)
	plt.title('Go Player Rank Distribution')
	plt.show()





def get_game_length(path="/"):

	file = open(path, encoding="utf8", errors='ignore')
	lines = [line.strip() for line in file.readlines()]

	length = 0

	for line in lines:


		for i in range(len(line)):

			if line[i:i+3] == ";B[":
				length += 1

			if line[i:i+3] == ";W[":
				length += 1
	
	if length > 200:
		length = 200

	return length

def get_game_length_dist(path="./data/ogs_games/"):

	file_paths = listdir(path)

	length_dist = [0]*201

	for game_path in tqdm(file_paths, "getting length dist: "):

		length = get_game_length(path=path+game_path)
		length_dist[length] += 1

	return length_dist


def plot_length_dist(path="./data/ogs_games/"):

	length_dist = get_game_length_dist(path=path)

	plt.bar(list(range(1, len(length_dist)+1)), length_dist)
	plt.ylabel('# of Games')
	plt.xlabel('Game Length')
	plt.xticks(fontsize=6)
	plt.title('Game Length Distribution')
	plt.show()




