import pygame
import numpy as np

class GoGUI(object):

    def __init__(self, board_width=600, board_height=600):

        # init pygame
        pygame.init()
        
        # load assets
        self.load_assets()

    def draw_boad(self):
        pass

    def draw_pieces(self):
        pass

    def play_sound(self):
        pass

    def load_assets(self):

        # load board image
        self.board_img = pygame.image.load("assets//background_img.jpg")

        # load white stones
        self.white_pieces = {}
        for i in range(16):
            self.white_pieces["W"+str(i)] = pygame.image.load("assets/w{}.png".format(i))

        # load black stones
        self.black_pieces = {}
        self.white_pieces["B"+str(0)] = pygame.image.load("assets/b{}.png".format(0))

        # load sounds
        self.stone_sound = pygame.

    def human_vs_ai(self, ai):
        pass




def main():
    gg = GoGUI()


if __name__ == "__main__":
    main()
