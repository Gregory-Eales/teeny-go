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

class GoScraper(object):

    def __init__(self):

        self.base_url = 'https://online-go.com'
        self.client_id = "4vMV9BvynUsCvEu7fCFe8t0jbOuHDmSGCc7tVQk8"
        s1 = "UfeTVvZx5NflNufG0w5oKnD9l5MMhIwQZWxoROdbVRIBkEDKqQ9wPysHgdiKoKjit"
        s2 = "qlaWmsKOkqMWQ8hZmqdntCT4Nc3ZgMx9J00d1yT5n44jVPzMkYGhONvHOPzC7Gh"
        self.client_secret = s1 + s2
        self.user_links = []
        self.dan_user_links = []
        self.dan_game_ids = []
        self.driver = webdriver.Safari()

    def check_connection(self):
        response = requests.get(self.base_url)
        if response.status_code == 200:return True
        else: return False

    def get_leaderboard_users(self):

        url= 'https://online-go.com/leaderboards'
        self.driver.get(url)
        time.sleep(1)
        content = self.driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(content,"html.parser")
        officials = soup.findAll("a",{"class":"Player"})

        for entry in officials:
            try:
                if self.base_url+str(entry['href']) not in self.user_links:
                    self.user_links.append(self.base_url+str(entry['href']))
            except:None

    def get_users(self, url):

        self.driver.get(url)
        time.sleep(1)
        self.save_user_links()
        content = self.driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(content,"html.parser")
        officials = soup.findAll("a",{"class":"Player"})

        for entry in officials:
            try:
                if self.base_url+str(entry['href']) not in self.user_links:
                    self.user_links.append(self.base_url+str(entry['href']))
            except:None

    def save_user_links(self):

        file = open("./data/ogs_user_links/user_links.txt", 'w')
        for link in self.user_links:
            file.write(link + "\n")

        file.close()

    def save_dan_user_links(self):

        file = open("./data/ogs_user_links/dan_user_links.txt", 'w')
        for link in self.dan_user_links:
            file.write(link + "\n")

        file.close()

    def read_user_links(self):
        file = open("./data/ogs_user_links/user_links.txt", 'r')
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i][0:-1]

        self.user_links = lines

    def read_dan_user_links(self):
        file = open("./data/ogs_user_links/dan_user_links.txt", 'r')
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i][0:-1]

        self.dan_user_links = lines

    def scrape_users(self):
        self.get_leaderboard_users()
        for i in range(10):
            for url in self.user_links:
                print(len(self.user_links))
                self.get_users(url)


    def scrape_games(self):

        for link in self.user_links:
            self.driver.get(link)


    def get_dan_users(self):
        self.read_user_links()
        for user_link in self.user_links:
            try:
                if self.check_if_dan(user_link) and user_link not in self.dan_user_links:
                    self.dan_user_links.append(user_link)
                    self.save_dan_user_links()
                    print(len(self.dan_user_links))
            except: print("could not access")

    def check_if_dan(self, link):
        self.driver.get(link)
        time.sleep(2)
        content = self.driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(content,"html.parser")
        rank = soup.findAll("span",{"class":"Player-rank"})[0]

        if str(rank).split("]")[0][-1] == "d": return True
        else: return False

    def download_game(self, game_id):
        link = "https://online-go.com/api/v1/games/{}/sgf".format(game_id)
        r = requests.get(link, allow_redirects=True)
        open('data/ogs_dan_games/ogs_{}.sgf'.format(game_id), 'wb').write(r.content)


    def download_all_games(self):
        self.read_game_ids()
        downloaded_games = self.get_downloaded_games()
        for i in tqdm(range(len(self.dan_game_ids))):
            if str(self.dan_game_ids[i]) not in downloaded_games:
                self.download_game(self.dan_game_ids[i])

    def get_all_game_ids(self):

        self.read_dan_user_links()
        self.read_game_ids()

        num = len(self.dan_user_links)

        for i in tqdm(range(200, num)):
            user_id = self.dan_user_links[i].split("/")[4]
            self.dan_game_ids += self.get_game_ids(user_id)
            self.save_game_ids()

        self.save_game_ids()

    def save_game_ids(self):
        file = open('./data/ogs_user_links/ids.txt', "w")
        for id in self.dan_game_ids:
            file.write(id+"\n")
        file.close()

    def read_game_ids(self):
        file = open("./data/ogs_user_links/ids.txt", 'r')
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i][0:-1]

        self.dan_game_ids = lines

    def get_game_ids(self, user_id):

        start = "https://online-go.com/api/v1/players/"
        end = "/games/?page_size=100&page=1&source=play&ended__isnull=false&ordering=-ended"
        link = start + user_id + end

        game_ids = []
        searching = True

        while searching:
            r = requests.get(link)
            for game in r.json()["results"]:
                if game["width"] == 9 and game["height"]==9:
                    if str(game["id"]) not in game_ids:
                        game_ids.append(str(game["id"]))

            link = r.json()["next"]
            if type(link) != str:
                   searching = False


        return game_ids

    def get_downloaded_games(self):
        import glob
        ids = []
        paths = glob.glob('data/ogs_dan_games/*.sgf')
        for p in paths:
            ids.append(p.split("/")[-1].split(".")[0].split("_")[-1])

        return ids

def main():

    gs = GoScraper()
    gs.get_leaderboard_users()
    print(len(gs.user_links))
    gs.scrape_users()
    print(len(gs.user_links))

if __name__ == "__main__":
    main()
