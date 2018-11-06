import numpy as np
import pygame

class Space(object):

    def __init__(self):
        self.coordinates = None
        self.state = 0


class GoEngine(object):

    def __init__(self, graphics=True):

        self.board = np.zeros([9, 9, 3])
        self.turn = "white"
        # initiate graphics
        if graphics:
            pygame.init()
            self.screen = pygame.display.set_mode([720, 720])
            pygame.display.set_caption("GoEngine")
            self.clock = pygame.time.Clock()
            self.background = pygame.image.load("Oak.jpg")
            self.grid_length = 80
            self.letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
            self.spaces = {}
            for i in range(9):
                for letter in self.letters:
                    self.spaces[letter+str(i)] = Space()

            for i in range(9):
                for j in range(9):
                    self.spaces[self.letters[i]+str(j)].coordinates = [40 + j*80, 40 + i*80]




    def play(self):
        done = False
        self.turn = "white"
        while not done:
            position = 1
            # --- Main event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

                if event.type == pygame.MOUSEBUTTONDOWN:
                    position = pygame.mouse.get_pos()

            # --- Game logic should go here
            if position != 1:
                print(position)
                position = self.round_to_location(position)
                print(position)
                if position is not None:
                    if self.spaces[position].state == 0:
                        if self.turn == "white":
                            self.turn = "black"
                            print("its blacks turn")
                            self.spaces[position].state = 1
                        elif self.turn == "black":
                            self.turn = "white"
                            print("its whites turn")
                            self.spaces[position].state = -1


            # --- Screen-clearing code goes here

            # Here, we clear the screen to white. Don't put other drawing commands
            # above this, or they will be erased with this command.

            # If you want a background image, replace this clear with blit'ing the
            # background image.
            self.screen.fill([0, 0, 0])

            # --- Drawing code should go here
            self.draw_background()
            self.draw_pieces()
            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self.clock.tick(60)

        # Close the window and quit.
        pygame.quit()

    def draw_background(self):

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

    def draw_pieces(self):

        for elmnt in self.spaces:

            if self.spaces[elmnt].state == 1:
                pygame.draw.circle(self.screen, [255, 255, 255], self.spaces[elmnt].coordinates, 30, 0)

            if self.spaces[elmnt].state == -1:
                pygame.draw.circle(self.screen, [0, 0, 0], self.spaces[elmnt].coordinates, 30, 0)









