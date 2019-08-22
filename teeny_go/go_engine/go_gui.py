import pygame
from go_engine import GoEngine


class GoGUI(object):

    def __init__(self):

        # initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode([720, 720])
        pygame.display.set_caption("GoEngine")
        self.clock = pygame.time.Clock()
        self.background = pygame.image.load("assets/images/Oak.jpg")
        self.grid_length = 80
        self.stone_sound1 = pygame.mixer.Sound("assets/Go-Stone-Sound/stone1.wav")
        self.GoEngine = GoEngine()
        self.background_img = pygame.image.load("assets/images/background_img.jpg")
        self.background_img = pygame.transform.scale(self.background_img, [720, 720])

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
        pass

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
def main():
    go = GoGUI()
    go.title_screen()

if __name__ == "__main__":
    main()
