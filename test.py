import requests
import urllib.request
import time
import requests
from tqdm import tqdm
from multiprocessing import Process

def save_ids(ids):

	file = open('./data/10k_min_bot_game_ids.txt', "w")
	for id in ids:
		file.write(id+"\n")
	file.close()

base_url = "https://online-go.com/api/v1/players/{}/games/"
options_url= "?ended__isnull=false&ordering=-ended&page={}&page_size=100&source=play&width=9"

url = base_url + options_url

game_ids = []

for bot_id in ["168480", "677246", "820843", "65706", "60551", "559323", "752554", "851346", "374985", "490132", "575693", "802383", "818667", "633047", "585357", "298756", "780891", "845032"]:

	r = requests.get(url.format(bot_id, 1))
	count = r.json()["count"]//100 + 1

	for i in tqdm(range(count)):

		try:

			r = requests.get(url.format(bot_id, i+1))
			for game in r.json()["results"]:
				if game["width"] == 9 and game["height"]==9:

					black_rank = game["players"]["black"]["ranking"]
					white_rank = game["players"]["white"]["ranking"]

					if white_rank > 15 or black_rank > 15:
						game_ids.append(str(game["id"]))
		except:
			break						

save_ids(game_ids)
print("valid games: {}".format(len(game_ids)))