import time
import sys
import requests

from utils.scraper import GoScraper

def main(argv):

   # load data with arg options

   scraper = GoScraper()

   #scraper.get_player_ids()

   scraper.read_player_ids()

   #print("Game IDs:",len(scraper.dan_game_ids))
   print("Player IDs:", len(scraper.player_ids))

   scraper.get_all_game_ids()
   
   #scraper.download_dan_games()

if __name__ == "__main__":
   main(sys.argv)


