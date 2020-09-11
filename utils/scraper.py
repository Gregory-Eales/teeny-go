import requests
import urllib.request
import time
import requests
from tqdm import tqdm
from multiprocessing import Process
import random
from fp.fp import FreeProxy



class GoScraper(object):


	def __init__(self):

		self.base_url = 'https://online-go.com'

		self.player_ids = []
		self.player_ids = []

		self.game_ids = []

		self.message = ""

	def check_connection(self):
		response = requests.get(self.base_url)
		if response.status_code == 200:return True
		else: return False


	def check_proxies(self):

		to_remove = []

		for p in tqdm(self.proxies):

			try:

				proxy = { 
		              "http"  : "http://{}".format(p), 
		              "https" : "https://{}".format(p)
		            }
				r = requests.get(link,
					allow_redirects=True,
					proxies=proxy,
					headers={'User-Agent': 'Chrome'},
					verify=False,
					timeout=1)

			except:
				to_remove.append(p)


		for p in to_remove:
			self.proxies.remove(p)


		print("valid proxeis: {}".format(len(self.proxies)))


	def get_player_ids(self, save=True):

		try: self.read_player_ids()

		except: print("warning: unable to load existing player ids")

		link = "https://online-go.com/api/v1/players/?page_size=100&ranking={}&page={}"


		for rank in range(20, 40):
			# get total players
			r = requests.get(link.format(rank, 1))
			total_players = r.json()['count']

			p_bar = tqdm(range(total_players//100 + 1))
			ids_found = 0
			failures = 0

			for i in p_bar:

				if i // 10 and save:
					self.save_player_ids()

				r = requests.get(link.format(rank, i+1))
				try:
					for player in r.json()['results']:
						if player['ranking'] >= 20:
							ids_found += 1

							if player['id'] not in self.player_ids:
								self.player_ids.append(player['id'])
				except:
					failures += 1

				p_bar.set_postfix({'players found': ids_found, "failures":failures})


			print("Rank {}d | # Players: {} | Failures: {}".format(rank-29, ids_found, failures))

		self.save_player_ids()

	def save_player_ids(self):

		file = open("./data/player_ids.txt", 'w')
		for player_id in self.player_ids:
			file.write(str(player_id) + "\n")

		file.close()

	def read_player_ids(self):
		file = open("./data/player_ids.txt", 'r')
		lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i][0:-1]

		self.player_ids = lines


	def download_game(self, game_id):
		link = "https://online-go.com/api/v1/games/{}/sgf".format(game_id)

		proxy = { 
              "http"  : "http://{}".format(self.proxies[10]), 
              "https" : "https://{}".format(self.proxies[10])
            }
		r = requests.get(link,
			allow_redirects=True,
			proxies=proxy,
			headers={'User-Agent': 'Chrome'},
			verify=False)

		open('data/ogs_games/ogs_{}.sgf'.format(game_id), 'wb').write(r.content)


	def download_games(self):
		self.proxies = FreeProxy().get_proxy_list()
		self.check_proxies()
		self.read_game_ids()
		random.shuffle(self.game_ids)
		for i in tqdm(range(len(self.game_ids[0:100]))):
			self.download_game(self.game_ids[i])
			time.sleep(0.6)

	def threaded_download_games(self, num_processes=24):

		self.read_game_ids()

		self.game_ids.reverse()

		split = int(len(self.game_ids)//num_processes)
		processes = [i for i in range(num_processes)]

		print(split)
		print(processes)
		print(len(self.game_ids[(0)*split:(0+1)*split]))

		for p in range(num_processes):
			processes[p] = Process(
				target=self.game_download_process,
				args=(self.game_ids[(p)*split:(p+1)*split], )
				)	

			processes[p].start()
		
		for p in range(num_processes):
			processes[p].join()

	def game_download_process(self, ids):

		for game_id in tqdm(ids):
			
			self.download_game(game_id)
			time.sleep(0.2)


	def get_all_game_ids(self):

		self.read_player_ids()

		random.shuffle(self.player_ids)


		p_bar = tqdm(self.player_ids)

		for player_id in p_bar:
			self.game_ids += self.get_game_ids(player_id)
			self.save_game_ids()
			p_bar.set_postfix({'games': len(self.game_ids),'message':self.message})

		self.save_game_ids()

	def save_game_ids(self):
		file = open('./data/game_ids.txt', "w")
		for id in self.game_ids:
			file.write(id+"\n")
		file.close()

	def read_game_ids(self):
		file = open("./data/game_ids.txt", 'r')
		lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i][0:-1]

		self.game_ids = lines

	def get_game_ids(self, player_id):

		start = "https://online-go.com/api/v1/players/"
		end = "/games/?page_size=100&page=1&source=play&ended__isnull=false&ordering=-ended"
		link = start + player_id + end + "&width=9"

		game_ids = []
		searching = True

		while searching:

			try:

				counter = 0
				while True:
					r = requests.get(link, timeout=5.0)
					if 'detail' in r.json().keys():

						if r.json()['detail'] == 'Request was throttled.':
							self.message = "throttled"
							time.sleep(1)

					else:
						self.message = "un-throttled"
						break

					counter+=1

					if counter > 10:
						"looped"
						break

				for game in r.json()["results"]:
					if game["width"] == 9 and game["height"]==9:
						
						black_rank = game["players"]["black"]["ranking"]
						white_rank = game["players"]["white"]["ranking"]

						if white_rank >= 20 and black_rank >= 20:
							if str(game["id"]) not in game_ids:
								game_ids.append(str(game["id"]))

				link = r.json()["next"]
				if type(link) != str:
					   searching = False

			
			except: searching = False

		return game_ids




if __name__ == "__main__":

	#scraper = GoScraper()

	print(FreeProxy().get_proxy_list())

	#scraper.get_players()

	#t = time.time()
	
	#r = requests.get("https://online-go.com/api/v1/players/273312/games/?page_size=100&page=1&source=play&ended__isnull=false&ordering=-ended&width=9", timeout=1.0, allow_redirects=True)
	#print(r.json())
	#print(time.time()-t)