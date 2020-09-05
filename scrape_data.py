from utils.sgf_scraper import GoScraper
import time
import sys

def main(argv):

   # load data with arg options

   # load / init model with arg options

   # train loop:
   #	
   #	1. for batch:
   #		1. make prediction
   #		2. calculate error
   #		3. update weights
   #
   #	2. test model Elo
   # 	3. save metrics

   gs = GoScraper()
   gs.download_all_games()

if __name__ == "__main__":
   main(sys.argv)


