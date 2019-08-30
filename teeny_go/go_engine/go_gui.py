import pygame
from cython_go_engine import GoEngine
import time
import timeit
import numpy as np

class Space(object):
    """ space Object for storing state and location __init__ """
    def __init__(self):
        """ space Object for storing state and location """
        self.coordinates = None
        self.state = 0


class GoGUI(object):

    def __init__(self):
        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.spaces = {}
        for i in range(9):
            for letter in self.letters:
                self.spaces[letter + str(i)] = Space()

        for i in range(9):
            for j in range(9):
                self.spaces[self.letters[i] + str(j)].coordinates = [40 + j * 80, 40 + i * 80]
        # initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode([720, 720])
        pygame.display.set_caption("GoEngine")
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load("assets/images/Oak.jpg")
        self.grid_length = 80
        self.stone_sound1 = pygame.mixer.Sound("assets/Go-Stone-Sound/stone1.wav")
        self.background_img = pygame.image.load("assets/images/background_img.jpg")
        self.background_img = pygame.transform.scale(self.background_img, [720, 720])

        self.GoEngine = GoEngine()

    def run(self):
        pass

    def title_screen(self):
        pygame.display.flip()
        myfont = pygame.font.SysFont('Comic Sans MS', 80)
        myfont2 = pygame.font.SysFont('Comic Sans MS', 60)
        textsurface = myfont.render('Teeny Go', False, (255, 255, 255))
        textsurface2 = myfont2.render('Play', False, (255, 255, 255))
        textsurface3 = myfont2.render('Play', False, (0, 0, 0))
        quited = False
        done = False
        while not done:
            position = pygame.mouse.get_pos()
            # get inputs from user
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    quited = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    position = pygame.mouse.get_pos()
                    done=True
            self.draw_background()
            #self.image.save(self.screen, "background_img.jpg")
            pygame.draw.rect(self.screen, [0, 0, 0], [190, 180, 340, 80], 0)
            self.screen.blit(textsurface,(230, 195))
            if position[0] > 285 and position[0] < 435:
                if position[1] > 350 and position[1] < 410:
                    pygame.draw.rect(self.screen, [255, 255, 255], [285, 350, 150, 60], 0)
                    self.screen.blit(textsurface3,(315, 360))
                else:
                    pygame.draw.rect(self.screen, [0, 0, 0], [285, 350, 150, 60], 0)
                    self.screen.blit(textsurface2,(315, 360))
            else:
                pygame.draw.rect(self.screen, [0, 0, 0], [285, 350, 150, 60], 0)
                self.screen.blit(textsurface2,(315, 360))
            pygame.display.flip()
        return quited

    def play(self):

        pygame.display.flip()

        while self.GoEngine.is_playing:
            type_for_capture = 0
            position = 1
            # get inputs from user
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    position = pygame.mouse.get_pos()

            if position != 1:
                # stand off is set to False
                stand_off = 0
                # translate position to one of the playable spots
                position = self.round_to_location(position)
                position = self.convert_to_num(position)

            ##############
            # Game Logic #
            ##############
            pygame.display.flip()
            #self.GoEngine.move = position
            move = position
            if move != 1:
                # check if move is valid

                move = [move[0], move[1]]

                if self.GoEngine.check_valid(move) == True:
                    pygame.display.flip()
                    pygame.mixer.Sound.play(self.stone_sound1)
                    #self.GoEngine.make_move(move)
                    print(self.GoEngine.get_invalid_moves().reshape([9, 9]))


            # fill blacks
            self.screen.fill([0, 0, 0])
            # draw
            self.draw_background()
            self.draw_pieces()
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
            # --- Limit to 60 frames per second
            self.clock.tick(60)

        # Close the window and quit.
        pygame.quit()

    def draw_background(self):

        if True:
            self.screen.blit(self.background_img, [0, 0])
        else:
            # fill background with oak image
            self.screen.blit(self.background, [0, 0])
            self.screen.blit(self.background, [400, 0])
            self.screen.blit(self.background, [0, 320])
            self.screen.blit(self.background, [400, 320])
            self.screen.blit(self.background, [0, 640])
            self.screen.blit(self.background, [400, 640])
            # draw circle markers
            pygame.draw.circle(self.screen, [0, 0, 0], [200, 200], 8, 0)
            pygame.draw.circle(self.screen, [0, 0, 0], [520, 200], 8, 0)
            pygame.draw.circle(self.screen, [0, 0, 0], [200, 520], 8, 0)
            pygame.draw.circle(self.screen, [0, 0, 0], [520, 520], 8, 0)
            # draw gridlines
            for i in range(9):
                pygame.draw.line(self.screen, [0, 0, 0], [40, 40 + self.grid_length*i],  [680, 40+self.grid_length*i])
                pygame.draw.line(self.screen, [0, 0, 0], [40 + self.grid_length * i, 40], [40 + self.grid_length * i, 680])
            pygame.display.flip()
            pygame.image.save(self.screen, "background_img.jpg")

    def draw_pieces(self):

        for i in range(9):
            for j in range(9):
                coordinates = [40 + j * 80, 40 + i * 80]
                if self.GoEngine.board[i][j] == 1:
                    pygame.draw.circle(self.screen, [0, 0, 0], coordinates, 30, 0)

                if self.GoEngine.board[i][j] == -1:
                    pygame.draw.circle(self.screen, [255, 255, 255], coordinates, 30, 0)

    def round_to_location(self, location):
        x, y = location[0], location[1]

        if x < 20 or x > 700:
            return None

        if y < 20 or y > 700:
            return None
        x = abs((x - 20) // 80)
        y = abs((y - 20) // 80)
        y = self.letters[y]

        return y+str(x)

    def convert_to_num(self, pos):
        y = int(self.letters.index(pos[0]))
        x = int(pos[1])
        return (x, y)

def main():
    go = GoGUI()
    go.play()

if __name__ == "__main__":
    print("succces")
    main()
