import time
import os
import random

import pygame
import numpy as np


class Viewer(object):

    def __init__(self, board_size=600):

        # save board dimensions
        self.board_size = board_size

        # init pygame
        pygame.init()
        self.screen = pygame.display.set_mode((board_size, board_size))
        pygame.display.set_caption('Go')

        # load assets
        self.load_assets()

        # create dict key lists
        self.sounds = list(self.sounds.keys())
        self.white_piece_list = list(self.white_pieces.keys())
        self.black_piece_list = list(self.black_pieces.keys())

    def load_assets(self):

        # get path
        path = os.path.dirname( os.path.realpath( __file__ ) )

        # load board image
        self.board_img = pygame.image.load(path+"/assets/images/background_img.jpg")
        self.board_img = pygame.transform.scale(self.board_img, [self.board_size, self.board_size])
        self.screen.blit(self.board_img, [0, 0])
        pygame.display.update()

        # load black stones
        self.black_pieces = {}
        self.black_pieces["B"+str(0)] = pygame.image.load(path+"/assets/images/b{}.png".format(0))
        self.black_pieces["B"+str(0)] = pygame.transform.scale(self.black_pieces["B"+str(0)],
         [int(self.board_size/10), int(self.board_size/10)])

        # set icon
        pygame.display.set_icon(self.black_pieces["B"+str(0)])

        # load white stones
        self.white_pieces = {}
        for i in range(16):
            self.white_pieces["W"+str(i)] = pygame.image.load(path+"/assets/images/w{}.png".format(i))
            self.white_pieces["W"+str(i)] = pygame.transform.scale(self.white_pieces["W"+str(i)],
             [int(self.board_size/10), int(self.board_size/10)])

        # load sounds
        self.sounds = {}
        for i in range(1,6):
            self.sounds["Stone"+str(i)] = pygame.mixer.Sound(path+"/assets/sound/stone{}.wav".format(i))


    def draw_board(self):
        self.screen.blit(self.screen, [0, 0])

    def draw_pieces(self):

        for i, piece in enumerate(self.pieces):
            if piece != None:
                screen.blit(piece, [(i)%9, (i)//9])

    def update_pieces(self):
        pass

    def play_stone_sound(self):
        pygame.mixer.Sound.play(self.sounds[random.choice(self.sounds)])

    def get_human_move(self):

        getting = True

        while getting:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    getting = False
                    pygame.quit()


                if event.type ==  pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()

    def human_vs_ai(self, ai):

        clock = pygame.time.Clock()
        crashed = False

        while not crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                    print("end")


            pygame.display.update()
            clock.tick(60)


def main():
    pass


if __name__ == "__main__":
    main()
