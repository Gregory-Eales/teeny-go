import requests
import urllib.request
import time
import requests
from tqdm import tqdm
from multiprocessing import Process
import random
#from fp.fp import FreeProxy



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
				r = requests.get("https://online-go.com/api/",
					allow_redirects=True,
					proxies=proxy,
					headers={'User-Agent': 'Chrome'},
					verify=False)

			except:
				to_remove.append(p)


		for p in to_remove:
			self.proxies.remove(p)


		print("valid proxeis: {}".format(len(self.proxies)))


	def get_player_ids(self, save=True):

		# try: self.read_player_ids()

		# except: print("warning: unable to load existing player ids")

		#link = "https://online-go.com/api/v1/players/?page_size=100&ranking={}&page={}"
		link = 'https://online-go.com/api/v1/ladders/315/players?page={}&page_size=100'
		p_bar = tqdm(range(14))
		failures = 0
		ids_found = 0
		for i in range(14):
			r = requests.get(link.format(i+1))
			try:
				for player in r.json()['results']:
					if player['player']['ratings']['overall']['rating'] > 1000:
						ids_found += 1
						if player['id'] not in self.player_ids:
							self.player_ids.append(player['id'])
			except Exception as e:
				print(e)
				failures += 1

			p_bar.set_postfix({'players found': ids_found, "failures":failures})


			print("# Players: {} | Failures: {}".format(ids_found, failures))

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


	def write_game(self, game_id, content):
		open('data/ogs-high-quality/ogs_{}.sgf'.format(game_id), 'wb').write(content)



	def download_game(self, game_id, proxy=""):
		link = "https://online-go.com/api/v1/games/{}/sgf".format(game_id)
		time.sleep(0.3)
		for i in range(3):

			r = requests.get(link, allow_redirects=True)

			content = r.content.decode("utf-8").split("\n")
		
			if len(content) > 10 or content[0]!="detail":
				self.write_game(game_id, r.content)
				return 0

			else:
				time.sleep(10)
		"""
		for i in range(1):

			try:


				if proxy != "":
					r = requests.get(link,
						allow_redirects=True,
						proxies=proxy)
				else:
					r = requests.get(link,
						allow_redirects=True)

				content = r.content.decode("utf-8").split("\n")

				if len(content) > 10:

					black_found = False
					white_found = False

					

					for line in content:

						if line[0:3] == "BR[":

							if line[-2] == "d":
								self.write_game(game_id, r.content)
								return 0

							elif int(line[3:-2]) <= 10:
								self.write_game(game_id, r.content)
								return 0

							black_found = True

						elif line[0:3] == "WR[":

							if line[-2] == "d":
								self.write_game(game_id, r.content)
								return 0

							elif int(line[3:-2]) <= 10:
								self.write_game(game_id, r.content)
								return 0

							white_found = True

						if white_found and black_found:
							return 0				
					

					self.write_game(game_id, r.content)
					
				break

			except:
				time.sleep(0.1)

			"""

	def download_games(self):
		#self.proxies = FreeProxy().get_proxy_list()
		#self.check_proxies()
		self.read_game_ids()
		random.shuffle(self.game_ids)
		for i in tqdm(range(len(self.game_ids))):
			self.download_game(self.game_ids[i])

	def threaded_download_games(self, num_processes=1):

		self.read_game_ids()

		random.shuffle(self.game_ids)

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
			time.sleep(0.1)


	def get_all_game_ids(self):

		self.read_player_ids()

		random.shuffle(self.player_ids)


		p_bar = tqdm(self.player_ids)

		for player_id in p_bar:
			try:
				self.game_ids += self.get_game_ids(player_id)
				self.save_game_ids()
			except:
				pass
			p_bar.set_postfix({'games': len(self.game_ids),'message':self.message})

		self.save_game_ids()

	def save_game_ids(self):
		file = open('./data/ogs-high-quality.txt', "w")
		for id in self.game_ids:
			file.write(id+"\n")
		file.close()

	def read_game_ids(self):
		file = open('./data/ogs-high-quality.txt', 'r')
		lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i][0:-1]

		self.game_ids = lines

	def get_game_ids(self, player_id):

		start = "https://online-go.com/api/v1/players/"
		end = "/games/?page_size=100&page=1"
		link = start + player_id + end + "&width=9"

		game_ids = []
		searching = True

		while searching:

			counter = 0
			while True:
				r = requests.get(link, timeout=5.0)
				if 'detail' in r.json().keys():

					if r.json()['detail'] == 'Request was throttled.':
						self.message = "throttled"
						time.sleep(5)

				else:
					self.message = "un-throttled"
					break

				counter+=1

				if counter > 10:
					break
			
			if 'results' not in r.json():
				print(r.json())			



			for game in r.json()["results"]:
				if game["width"] == 9 and game["height"]==9:
					
					black_rank = game["players"]["black"]["ratings"]["overall"]["rating"]
					white_rank = game["players"]["white"]["ratings"]["overall"]["rating"]

					if white_rank >= 1000 or black_rank >= 1000:
						if str(game["id"]) not in game_ids:
							game_ids.append(str(game["id"]))

			link = r.json()["next"]
			if type(link) != str:
					searching = False

		return game_ids




if __name__ == "__main__":

	#scraper = GoScraper()

	r = requests.get("https://online-go.com/api/v1/games/26823200/sgf")
	print(r.content.decode("utf-8").split("\n"))
	#scraper.get_players()

	#t = time.time()
	
	#r = requests.get("https://online-go.com/api/v1/players/273312/games/?page_size=100&page=1&source=play&ended__isnull=false&ordering=-ended&width=9", timeout=1.0, allow_redirects=True)
	#print(r.json())
	#print(time.time()-t)