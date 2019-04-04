from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
from matplotlib import pyplot as plt
import pygame


#create model
model = Sequential()
#add model layers
model.add(Conv2D(60, kernel_size=2, activation='tanh', padding="same", input_shape=(9,9,1)))
model.add(Conv2D(50, kernel_size=2, activation='tanh', padding="same"))
model.add(Conv2D(30, kernel_size=2, activation='tanh', padding="same"))

#model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
model.add(Flatten())
model.add(Dense(160, activation='tanh', use_bias=True))
model.add(Dense(81, activation='sigmoid', use_bias=True))


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.load_weights("model.h5")

prediction = model.predict(np.zeros([1, 9, 9, 1]))

maximum = prediction.max()

pred_truth = prediction == maximum

print(pred_truth.reshape(9, 9).astype(int))


class Space(object):
    """ space Object for storing state and location __init__ """
    def __init__(self):
        """ space Object for storing state and location """
        self.coordinates = None
        self.state = 0



class GoPlayerEngine(object):

    def __init__(self, graphics=True):
        self.turn = "white"
        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        self.spaces = {}
        self.black_score = 0
        self.white_score = 0
        for i in range(9):
            for letter in self.letters:
                self.spaces[letter + str(i)] = Space()

        for i in range(9):
            for j in range(9):
                self.spaces[self.letters[i] + str(j)].coordinates = [40 + j * 80, 40 + i * 80]

        # initiate graphics
        if graphics:
            pygame.init()
            self.screen = pygame.display.set_mode([720, 720])
            pygame.display.set_caption("GoEngine")
            self.clock = pygame.time.Clock()
            self.background = pygame.image.load("images/Oak.jpg")
            self.grid_length = 80
            self.stone_sound1 = pygame.mixer.Sound("stone1.wav")


    def init_keras_ai(self):
    	#create model
    	model = Sequential()
    	#add model layers
    	model.add(Conv2D(60, kernel_size=2, activation='tanh', padding="same", input_shape=(9,9,1)))
    	model.add(Conv2D(50, kernel_size=2, activation='tanh', padding="same"))
    	model.add(Conv2D(30, kernel_size=2, activation='tanh', padding="same"))
    	#model.add(Conv2D(40, kernel_size=2, activation='relu', padding="same"))
    	model.add(Flatten())
    	model.add(Dense(160, activation='tanh', use_bias=True))
    	model.add(Dense(81, activation='sigmoid', use_bias=True))
    	#compile model using accuracy to measure model performance
    	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    	model.load_weights("model.h5")
    	self.model = model

    def positions_to_array(self):

    	keys = self.spaces.keys()

    	board = []


    	for key in keys:
    		board.append(self.spaces[key].state)




    	return np.array(board).reshape(1, 9, 9, 1)


    	

    def make_ai_move(self):
    	

    	prediction = model.predict(self.positions_to_array())

		maximum = prediction.max()

		pred_truth = prediction == maximum

		print(pred_truth.reshape(9, 9).astype(int))


    def opening_screen(self):
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
        if not self.opening_screen():
            done = False
        self.turn = "white"
        while not done:
            type_for_capture = 0
            position = 1

            if turn == "black":
            	position = self.make_ai_move()
            	pygame.mixer.Sound.play(self.stone_sound1)

            else:
	            # get inputs from user
	            for event in pygame.event.get():
	                if event.type == pygame.QUIT:
	                    done = True

	                if event.type == pygame.MOUSEBUTTONDOWN:
	                    position = pygame.mouse.get_pos()
	                    pygame.mixer.Sound.play(self.stone_sound1)

            # main game logic
            # if there is a click at a location

            if position != 1:
                # stand off is set to False
                stand_off = 0
                # translate position to one of the playable spots
                if turn == "white":
                	position = self.round_to_location(position)
                #if the position is good then...
                if position is not None:

                    # if the board space is empty...
                    if self.spaces[position].state == 0:
                        # place white or black depending on the turn
                        if self.turn == "white":
                            self.spaces[position].state = 1
                        elif self.turn == "black":
                            self.spaces[position].state = -1
                        # if
                        if self.check_capture_pieces(position) == "white" or self.check_capture_pieces(position) == "black":
                            if self.check_capture_pieces(position) == "white":
                                type_for_capture = 1
                                stand_off = 1

                            if self.check_capture_pieces(position) == "black":
                                type_for_capture = -1
                                stand_off = 1

                        if self.check_capture_pieces(position) == 0 or stand_off == 1:
                            if self.turn == "white":
                                self.turn = "black"
                                #print("its blacks turn")
                            elif self.turn == "black":
                                self.turn = "white"
                                #print("its whites turn")

                        elif self.check_capture_pieces(position) == 1:
                            print("Invalid Move")
                            self.spaces[position].state = 0



            if type_for_capture != 0:
                self.capture_pieces(type_for_capture)


            # clear screen

            # Here, we clear the screen to white. Don't put other drawing commands
            # above this, or they will be erased with this command.

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

    def capture_pieces(self, type_for_capture):
        for i in range(9):
            for j in range(9):
                location_state = self.spaces[self.letters[i] + str(j)].state
                if location_state != 0 and location_state == type_for_capture:
                    group = self.get_group([i, j], location_state)
                    if group != []:
                        free = self.check_neighbors(group, location_state)
                        if free == "False":
                            self.remove_group(group)

    def check_capture_pieces(self, position):
        killing_itself = 0
        for i in range(9):
            for j in range(9):
                location_state = self.spaces[self.letters[i] + str(j)].state
                if location_state != 0:
                    group = self.get_group([i, j], location_state)
                    if group != []:
                        free = self.check_neighbors(group, location_state)
                        if free == "False":
                                if [self.letters.index(position[0]), int(position[1])] in group:
                                    killing_itself = 1
                                if location_state == 1 and self.spaces[position].state != 1:
                                    return "white"
                                if location_state == -1 and self.spaces[position].state != -1:
                                    return "black"

        return killing_itself
        return 0

    def check_neighbors(self, group, state_type):
        liberty = "False"

        for position in group:

            a, b = position[0], position[1]

            if a < 8:
                if self.spaces[self.letters[a+1]+str(b)].state == 0:
                    return True

            if a > 0:
                if self.spaces[self.letters[a-1]+str(b)].state == 0:
                    return True

            if b < 8:
                if self.spaces[self.letters[a] + str(b+1)].state == 0:
                    return True

            if b > 0:
                if self.spaces[self.letters[a]+str(b-1)].state == 0:
                    return True

        return liberty


    def get_group(self, position, state_type):
        stone_group = []
        stone_group.append(position)
        for j in range(40):
            for pos in stone_group:
                a, b = pos[0], pos[1]
                if a > 0:
                    if self.spaces[self.letters[a-1] + str(b)].state == state_type and [a-1, b] not in stone_group:
                        stone_group.append([a-1, b])

                if a < 8:
                    if self.spaces[self.letters[a+1] + str(b)].state == state_type and [a+1, b] not in stone_group:
                        stone_group.append([a+1, b])

                if b > 0:
                    if self.spaces[self.letters[a] + str(b-1)].state == state_type and [a, b-1] not in stone_group:
                        stone_group.append([a, b-1])

                if b < 8:
                    if self.spaces[self.letters[a] + str(b+1)].state == state_type and [a, b+1] not in stone_group:
                        stone_group.append([a, b+1])

        return stone_group

    def remove_group(self, group):
        print(group)
        if self.spaces[self.letters[group[0][0]] + str(group[0][1])].state == 1:
            self.black_score = self.black_score + len(group)
        if self.spaces[self.letters[group[0][0]] + str(group[0][1])].state == -1:
            self.white_score = self.white_score + len(group)
        print("White Score: " + str(self.white_score))
        print("Black Score: " + str(self.black_score))
        for elmnt in group:
            self.spaces[self.letters[elmnt[0]] + str(elmnt[1])].state = 0


go_engine = GoPlayerEngine()

print(go_engine.positions_to_array())





