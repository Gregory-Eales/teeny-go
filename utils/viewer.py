import time
import os
import random

import pygame
import numpy as np
import torch
import pyspiel


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

        # initialize board
        self.initialize_board()

        # create dict key lists
        self.sounds_list = list(self.sounds.keys())
        self.white_piece_list = list(self.white_pieces.keys())
        self.black_piece_list = list(self.black_pieces.keys())

        # create move map
        self.move_map = self.get_move_map()

        # gui tracking
        self.stone_state_holder = []
        self.stone_graphics_holder = []
        for i in range(81): self.stone_state_holder.append(0)
        for i in range(81): self.stone_graphics_holder.append(None)

    def get_move_map(self):
        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = pyspiel.load_game("go", board_size)
        state = game.new_initial_state()
        return state.legal_actions()

    def initialize_board(self):
        # create go board
        board_size = {"board_size": pyspiel.GameParameter(9)}
        game = pyspiel.load_game("go", board_size)
        self.board_state = game.new_initial_state()
        self.game_states = []
        for i in range(7): self.game_states.append(np.zeros([9,9]))

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

    def generate_state_tensor(self):

        black = []
        white = []
        turn = self.board_state.current_player()

        if turn == 1:
            turn = [np.zeros([1, 9, 9])]

        elif turn == 0:
            turn = [np.ones([1, 9, 9])]

        for i in range(1, 6):
            black.append(np.copy(np.where(self.game_states[-i] == 1, 1, 0).reshape(1, 9, 9)))
            white.append(np.copy(np.where(self.game_states[-i] == -1, 1, 0).reshape(1, 9, 9)))

        black = np.concatenate(black, axis=0)
        white = np.concatenate(white, axis=0)
        turn = np.concatenate(turn, axis=0)

        output = np.concatenate([black, white, turn]).reshape(1, 11, 9, 9)

        return output

    def draw_board(self):
        self.screen.blit(self.board_img, [0, 0])

    def draw_pieces(self):

        constant = 7/120
        scaler = (1.00-(2*constant))*self.board_size/8.00

        for i, piece in enumerate(self.stone_graphics_holder):
            if piece != None:
                x = (i%9)*scaler + constant*self.board_size - self.board_size/20.00
                y = (i//9)*scaler + constant*self.board_size - self.board_size/20.00
                self.screen.blit(piece, [x, y])

    def update_board(self):
        state = self.board_state.observation_as_normalized_vector()
        state = np.array(state).reshape(-1, 81)
        state = (state[0] + state[1]*-1)
        self.game_states.append(np.copy(state.reshape(1, 9, 9)))
        state = state.tolist()



        for i, space in enumerate(state):

            # if the state has changed
            if space != self.stone_state_holder[i]:

                if space == -1:
                    self.stone_graphics_holder[i] = self.get_white_stone_img()

                elif space == 1:
                    self.stone_graphics_holder[i] = self.get_black_stone_img()

                else:
                    self.stone_graphics_holder[i] = None

        self.stone_state_holder = state

    def get_white_stone_img(self):
        return self.white_pieces[random.choice(self.white_piece_list)]

    def get_black_stone_img(self):
        return self.black_pieces[self.black_piece_list[0]]

    def play_stone_sound(self):
        pygame.mixer.Sound.play(self.sounds[random.choice(self.sounds_list)])

    def get_human_move(self):

        getting = True

        while getting:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    getting = False
                    pygame.quit()

                if event.type ==  pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()

                    constant = 7/120
                    scaler = (1.00-(2*constant))*self.board_size/8.00

                    x = (pos[0] - constant*self.board_size)/scaler
                    y = (pos[1] - constant*self.board_size)/scaler

                    pos = [int(round(x, 0)), int(round(y, 0))]
                    print(pos)

                    pos = self.move_map[pos[0]+pos[1]*9]
                    if pos in self.board_state.legal_actions():
                        self.board_state.apply_action(pos)
                        getting = False
                    else:
                        print("invalid move")

    def get_ai_move(self, ai, value_net):

        # get move tensor
        state_tensor = self.generate_state_tensor()
        state_tensor = torch.from_numpy(state_tensor).float()
        move_tensor = ai.forward(state_tensor)
        move_tensor = move_tensor.detach().numpy().reshape(-1)

        cl = value_net.forward(state_tensor)

        # remove invalid moves
        valid_moves = self.board_state.legal_actions_mask()
        valid_moves = np.array(valid_moves[0:441]).reshape(21, 21)
        valid_moves = valid_moves[1:10,1:10].reshape(81)
        valid_moves = np.append(valid_moves, 0)
        move_tensor = move_tensor * valid_moves

        moves = list(range(82))
        sum = np.sum(move_tensor[0:82])

        print("Confidence Level: {}".format(cl))

        if sum > 0:
            move = moves[np.argmax(move_tensor[0:82])]
            print("move:", move)
            #move = 9*(move%9) + move//9
            #move = np.random.choice(moves, p=move_tensor[0:82]/sum)
        else:
            print("ai: passed")
            move = 81


        #self.logger.INFO("AI moved at: {}".format(move))

        self.board_state.apply_action(self.move_map[int(move)])

    def human_vs_ai(self, ai, value_net):

        # initialize game variables
        clock = pygame.time.Clock()
        playing = True
        self.update_board()
        while playing:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    playing = False

            # make human move
            self.get_human_move()

            # play stone sound
            self.play_stone_sound()

            # update board from engine
            self.update_board()

            # draw board
            self.draw_board()

            # draw pieces
            self.draw_pieces()

            # update screen
            pygame.display.update()
            clock.tick(60)

            # make ai move()
            self.get_ai_move(ai, value_net)

            # play stone sound
            self.play_stone_sound()

            # update board from engine
            self.update_board()

            # draw board
            self.draw_board()

            # draw pieces
            self.draw_pieces()

            # update screen
            pygame.display.update()
            clock.tick(60)


def main():
    pass


if __name__ == "__main__":
    main()
