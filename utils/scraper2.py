import requests
import time
from tqdm import tqdm
"""

Get Player Stats:
	https://online-go.com/termination-api/players?ids=107699

Get Player Games:
	https://online-go.com/termination-api/player/107699/glicko2-history?speed=overall&size=9

Get Players of a Rank:
	https://online-go.com/api/v1/players/?page_size=100&ranking={}&page={}

1. use APIv1 to collect mutiple players at each rank

2. use termination-api to save other players of a certain rank from the list
   of 9x9 games played by initially collected

3. use termination-api to collect games from each player, recording
   using elo ranking criteria

"""

api_base = "https://online-go.com/api/v1/players/"

games_end= "/games/?page_size=1&page=1&width=9"

game_number_url = api_base + "{}"+ games_end


play_rank_url = api_base + "?page_size=100&ranking={}&page={}"


def get_number_games(player_id):

	r = requests.get(game_number_url.format(player_id))
	return r.json()['count']

def get_players_of_rank(rank):
	"""
	ranking goes from 1-40 (30k-9d)
	"""
	player_ids = []
	r = requests.get(play_rank_url.format(rank, 1))

	# page count for 100 players per page
	page_count = r.json()['count'] 

	print(page_count)

	for page in tqdm(range(1, page_count+1)):

		for player in r.json()['results']:
			player_ids.append(player['id'])
		
		time.sleep(1)
				
	return player_ids

def get_dan_players():

	player_ids = []
	for rank in range(30, 38):
		player_ids += get_players_of_rank(rank)

	return player_ids

def write_player_ids(player_ids, filename="player_ids.txt"):

	file = open(filename, "w")

	for id in player_ids:
		file.write(str(id)+"\n")


def main():
	player_ids = get_dan_players()
	write_player_ids(player_ids, filename="player_ids.txt")


if __name__ == "__main__":
	main()