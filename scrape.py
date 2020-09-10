import time
import sys
import requests

from utils.scraper import GoScraper

def main(argv):

   # load data with arg options

   scraper = GoScraper()



   scraper.read_dan_player_ids()

   print("Player IDs:",len(scraper.dan_player_ids))

   scraper.get_dan_player_ids()



   #scraper.read_dan_player_ids()

   #print("Game IDs:",len(scraper.dan_game_ids))
   #print("Player IDs:", len(scraper.dan_player_ids))

   #scraper.get_dan_player_ids()
   #scraper.save_dan_player_ids()
   #scraper.get_dan_game_ids()
   
   #scraper.download_dan_games()

if __name__ == "__main__":
   main(sys.argv)


