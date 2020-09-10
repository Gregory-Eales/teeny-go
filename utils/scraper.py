import requests
import urllib.request
import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import requests
from tqdm import tqdm
from multiprocessing import Process


class GoScraper(object):


	def __init__(self):

		self.base_url = 'https://online-go.com'

		self.player_ids = []
		self.dan_player_ids = []

		self.dan_game_ids = []

	def check_connection(self):
		response = requests.get(self.base_url)
		if response.status_code == 200:return True
		else: return False


	def get_dan_player_ids(self, save=True):

		link = "https://online-go.com/api/v1/players/?page_size=100&ranking={}&page={}"



		for rank in range(30, 40):
			# get total players
			r = requests.get(link.format(rank, 1))
			total_players = r.json()['count']

			p_bar = tqdm(range(total_players//100 + 1))
			ids_found = 0
			failures = 0

			for i in p_bar:

				if i // 10 and save:
					self.save_dan_player_ids()

				r = requests.get(link.format(rank, i+1))
				try:
					for player in r.json()['results']:
						if player['ranking'] >= 30:
							ids_found += 1

							if player['id'] not in self.dan_player_ids:
								self.dan_player_ids.append(player['id'])
				except:
					failures += 1

				p_bar.set_postfix({'dans found': ids_found, "failures":failures})


			print("Rank {}d | # Players: {} | Failures: {}").format(rank-29)

		self.save_dan_player_ids()

	def save_dan_player_ids(self):

		file = open("./data/dan_player_ids.txt", 'w')
		for player_id in self.dan_player_ids:
			file.write(str(player_id) + "\n")

		file.close()

	def read_dan_player_ids(self):
		file = open("./data/dan_player_ids.txt", 'r')
		lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i][0:-1]

		self.dan_player_ids = lines


	def download_game(self, game_id):
		link = "https://online-go.com/api/v1/games/{}/sgf".format(game_id)
		r = requests.get(link, allow_redirects=True)
		open('data/ogs_dan_games/ogs_{}.sgf'.format(game_id), 'wb').write(r.content)


	def download_dan_games(self):
		self.read_game_ids()
		for i in tqdm(range(len(self.dan_game_ids))):
			self.download_game(self.dan_game_ids[i])

	def threaded_download_dan_games(self, num_processes=24):

		self.read_game_ids()

		self.dan_game_ids.reverse()

		split = int(len(self.dan_game_ids)//num_processes)
		processes = [i for i in range(num_processes)]

		print(split)
		print(processes)
		print(len(self.dan_game_ids[(0)*split:(0+1)*split]))

		for p in range(num_processes):
			processes[p] = Process(
				target=self.game_download_process,
				args=(self.dan_game_ids[(p)*split:(p+1)*split], )
				)	

			processes[p].start()
		
		for p in range(num_processes):
			processes[p].join()

	def game_download_process(self, ids):

		for game_id in tqdm(ids):
			
			self.download_game(game_id)


	def get_dan_game_ids(self):

		self.read_dan_player_ids()


		p_bar = tqdm(self.dan_player_ids)

		for player_id in p_bar:
			self.dan_game_ids += self.get_game_ids(player_id)
			self.save_dan_game_ids()
			p_bar.set_postfix({'games': len(self.dan_game_ids)})

		self.save_dan_game_ids()

	def save_dan_game_ids(self):
		file = open('./data/dan_game_ids.txt', "w")
		for id in self.dan_game_ids:
			file.write(id+"\n")
		file.close()

	def read_game_ids(self):
		file = open("./data/dan_game_ids.txt", 'r')
		lines = file.readlines()
		for i in range(len(lines)):
			lines[i] = lines[i][0:-1]

		self.dan_game_ids = lines

	def get_game_ids(self, player_id):

		start = "https://online-go.com/api/v1/players/"
		end = "/games/?page_size=100&page=1&source=play&ended__isnull=false&ordering=-ended"
		link = start + player_id + end + "&width=9"

		game_ids = []
		searching = True

		while searching:

			try:
				r = requests.get(link)
				for game in r.json()["results"]:
					if game["width"] == 9 and game["height"]==9:
						if str(game["id"]) not in game_ids:
							game_ids.append(str(game["id"]))

				link = r.json()["next"]
				if type(link) != str:
					   searching = False

			except:
				searching = False

		return game_ids




if __name__ == "__main__":

	scraper = GoScraper()

	scraper.get_dan_players()