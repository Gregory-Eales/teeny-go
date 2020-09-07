from collections import namedtuple


class Point(namedtuple('Point', 'row col')):
	
	def neighbors(self):
		return [
			Point(self.row-1, self.col),
			Point(self.row + 1, self.col),
			Point(self.row, self.col - 1),
			Point(self.row, self.col + 1),
		]


class Player(object):

	black = 1
	white = 2
		
	@property
	def other(self):
		return Player.black if self == Player.white else Player.white



if __name__ == "__main__":

   pass