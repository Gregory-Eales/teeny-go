from utils.scraper import GoScraper
import time
import sys

def main(argv):

   # load data with arg options

   scraper = GoScraper()

   #scraper.get_dan_player_ids()
   #scraper.save_dan_player_ids()
   #scraper.get_dan_game_ids()
   
   scraper.threaded_download_dan_games()

if __name__ == "__main__":
   main(sys.argv)


