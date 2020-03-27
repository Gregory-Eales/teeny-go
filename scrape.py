from utils.sgf_scraper import GoScraper


gs = GoScraper()
gs.get_leaderboard_users()
print(len(gs.user_links))
gs.save_user_links()
print(len(gs.user_links))
