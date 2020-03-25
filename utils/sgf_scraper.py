import requests
import urllib.request
import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time
from bs4 import BeautifulSoup
from selenium import webdriver


class GoScraper(object):

    def __init__(self):

        self.base_url = 'https://online-go.com'
        self.client_id = "4vMV9BvynUsCvEu7fCFe8t0jbOuHDmSGCc7tVQk8"
        s1 = "UfeTVvZx5NflNufG0w5oKnD9l5MMhIwQZWxoROdbVRIBkEDKqQ9wPysHgdiKoKjit"
        s2 = "qlaWmsKOkqMWQ8hZmqdntCT4Nc3ZgMx9J00d1yT5n44jVPzMkYGhONvHOPzC7Gh"
        self.client_secret = s1 + s2
        self.user_links = []
        self.driver = webdriver.Safari()

    def check_connection(self):
        response = requests.get(self.base_url)
        if response.status_code == 200:return True
        else: return False

    def get_leaderboard_users(self):

        url= 'https://online-go.com/leaderboards'
        self.driver.get(url)
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
        content = self.driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(content,"html.parser")
        officials = soup.findAll("a",{"class":"Player"})

        for entry in officials:
            try:
                if self.base_url+str(entry['href']) not in self.user_links:
                    self.user_links.append(self.base_url+str(entry['href']))
            except:None

    def scrape_users(self):
        self.get_leaderboard_users()
        for i in range(10):
            for url in self.user_links:
                print(len(self.user_links))
                try:self.get_users(url)
                except: None

    def scrape_games(self):
        pass


def main():

    gs = GoScraper()
    gs.get_leaderboard_users()
    print(len(gs.user_links))
    gs.scrape_users()
    print(len(gs.user_links))

if __name__ == "__main__":
    main()
