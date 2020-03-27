import pygame
import numpy as np
import random
from enum import Enum
from snakeAgent import SnakeAgent
from keras.utils import to_categorical

class Action(Enum):
        Forward = 1
        Right = 2
        Left = 3

class Game:

    record = 0

    def __init__(self, cols, rows, size_snake, border_size, 
        border_color, background_color, food_color, snake_color,
        speed, max_steps):
        pygame.display.set_caption('Smart Snake')
        self.border_color = border_color
        self.background_color = background_color
        self.food_color = food_color
        self.snake_color = snake_color
        self.speed = speed
        self.max_steps = max_steps
        self.cols = cols
        self.rows = rows
        self.size_snake = size_snake
        self.border_size = border_size
        self.window_width = cols*self.size_snake + self.border_size*2
        self.window_height = rows*self.size_snake + self.border_size*2
        self.pygame_display = pygame.display.set_mode((self.window_width, self.window_height + 40))
        self.game_over = False
        self.player = Player(self)
        self.food = Food(self,self.player)
        self.score = 0
        self.eat_time = 0
        

    def __show_ui(self):
        font_bold = pygame.font.SysFont('arial', 15, True)

        black = pygame.Color('black')

        score_txt = font_bold.render('CURRENT SCORE:' , True, black)
        score_num_txt = font_bold.render(str(self.score), True, black)

        record_txt = font_bold.render('THE BEST SCORE: ', True, black)
        record_num_txt = font_bold.render(str(Game.record), True, black)

        self.__show_txt(score_txt, self.border_size, self.rows * self.size_snake + self.border_size * 2)
        self.__show_txt(score_num_txt, 160, self.rows * self.size_snake + self.border_size * 2)

        self.__show_txt(record_txt, self.border_size, self.rows * self.size_snake + self.border_size * 2 + 20)
        self.__show_txt(record_num_txt, 160, self.rows * self.size_snake + self.border_size * 2 + 20)
        
        self.__show_border()
        
        pt_x = self.window_width - self.border_size * 2 - self.size_snake
        pt_y = self.window_height + self.border_size

        arrow_pts = [[[pt_x,pt_y],[pt_x + self.size_snake//2, pt_y],[pt_x + self.size_snake, pt_y]], \
            [[pt_x, pt_y + self.size_snake//2],[pt_x + self.size_snake//2, pt_y + self.size_snake//2],[pt_x + self.size_snake, pt_y + self.size_snake//2]], \
            [[pt_x, pt_y + self.size_snake],[pt_x + self.size_snake//2, pt_y + self.size_snake],[pt_x + self.size_snake, pt_y + self.size_snake]]]
        
        if [self.player.x_speed,self.player.y_speed] == [1,0]:
            pygame.draw.polygon(self.pygame_display, self.food_color,[arrow_pts[0][0], arrow_pts[2][0], arrow_pts[1][2]])
        elif [self.player.x_speed,self.player.y_speed] == [-1,0]:
            pygame.draw.polygon(self.pygame_display, self.food_color,[arrow_pts[0][2], arrow_pts[2][2], arrow_pts[1][0]])
        elif [self.player.x_speed,self.player.y_speed] == [0,1]:
            pygame.draw.polygon(self.pygame_display, self.food_color,[arrow_pts[0][0], arrow_pts[0][2], arrow_pts[2][1]])
        elif [self.player.x_speed,self.player.y_speed] == [0,-1]:
            pygame.draw.polygon(self.pygame_display, self.food_color,[arrow_pts[2][0], arrow_pts[2][2], arrow_pts[0][1]])

    def __show_txt(self,txt, x, y):
        self.pygame_display.blit(txt,(x,y))

    def __show_border(self):
        pygame.draw.rect(self.pygame_display, self.border_color, [0,0, self.window_width, self.border_size])
        pygame.draw.rect(self.pygame_display, self.border_color, [self.window_width - self.border_size, 0, self.border_size, self.window_height])
        pygame.draw.rect(self.pygame_display, self.border_color, [0, self.window_height - self.border_size, self.window_width, self.border_size])
        pygame.draw.rect(self.pygame_display, self.border_color, [0, self.border_size, self.border_size, self.window_height - self.border_size - self.border_size])


    def show(self):
        self.pygame_display.fill(self.background_color)
        self.__show_ui()
        if not self.game_over:
            self.food.show(self)
            self.player.show(self)
        else:
            pygame.time.wait(300)
        pygame.display.update()
    
    def make_step(self, agent):
        state = agent.get_state(self)
        prediction = agent.get_prediction(state)
        action = to_categorical(np.argmax(prediction[0]), num_classes=3)
        self.player.move(action,self)
        if self.speed > 0:
            self.show()
            pygame.time.wait(self.speed)


    def make_train_step(self, agent, action=None):
        state_old = agent.get_state(self)
        if not action:
            if random.random() > agent.epsilon:
                prediction = agent.get_prediction(state_old)
                action = to_categorical(np.argmax(prediction[0]), num_classes=3)
            else:
                action = to_categorical(random.randint(0, 2), num_classes=3)
        self.player.move(action, self)
        state_new = agent.get_state(self)
        reward = agent.get_reward(self)
        agent.train_memory(state_old, state_new, action, reward, self.game_over)
        if self.speed > 0:
            self.show()
            pygame.time.wait(self.speed)

class Player(object):

    def __init__(self, game):
        self.x = game.cols // 2
        self.y = game.rows // 2
        self.position = []
        self.position.append([self.x, self.y])
        self.length = 1
        self.must_grow = False
        self.x_speed = 1
        self.y_speed = 0

    def move(self, action, game):
        move_array = [self.x_speed, self.y_speed]

        if self.must_grow:
            self.length = self.length + 1
            self.position.append([self.x, self.y])
            game.eat_time = 0
            self.must_grow = False

        act = 0
        if np.array_equal([1, 0, 0], action):
            act = Action.Forward
        elif np.array_equal([0, 1, 0], action):
            act = Action.Right
        elif np.array_equal([0, 0, 1], action):
            act = Action.Left
        # forward
        if act == Action.Forward:
            move_array = self.x_speed, self.y_speed
        # right
        elif act == Action.Right:

            if self.y_speed == 0:
                move_array = [0, self.x_speed]

            elif self.x_speed == 0:
                move_array = [-self.y_speed, 0]
        # left
        elif act == Action.Left:
            
            if self.y_speed == 0:
                move_array = [0, -self.x_speed]

            elif self.x_speed == 0:
                move_array = [self.y_speed, 0]

        self.x_speed = move_array[0]
        self.y_speed = move_array[1]
        self.x += self.x_speed
        self.y += self.y_speed

        if game.eat_time >= game.max_steps or self.x < 0 or self.x >= game.cols or self.y < 0 or self.y >= game.rows or [self.x, self.y] in self.position:
            game.game_over = True
        
        game.eat_time += 1
        if [self.x, self.y] == [game.food.x, game.food.y]:
            self.must_grow = True
            game.food.random_pos(game, self)
            game.score = game.score + 1
            if game.score > Game.record:
                Game.record = game.score

        if self.position[-1] != [self.x, self.y]:
            if self.length > 1:
                for i in range(0, self.length - 1):
                    self.position[i] = self.position[i + 1]
            self.position[-1] = [self.x, self.y]

    def show(self, game):
        for i in range(self.length):
            x_temp, y_temp = self.position[i]
            posX = x_temp * game.size_snake + game.border_size
            posY = y_temp * game.size_snake + game.border_size
            pygame.draw.rect(game.pygame_display, game.snake_color, [posX + 1,posY + 1, game.size_snake - 2, game.size_snake - 2])

class Food(object):

    def __init__(self,game,player):
        self.x = player.position[-1][0]+2
        self.y = player.position[-1][1]+2
        self.random_pos(game,player)

    def random_pos(self, game, player):
        self.x = random.randint(0, game.cols - 1)
        self.y = random.randint(0, game.rows - 1)
        if [self.x, self.y] in player.position:
            return self.random_pos(game,player)
        else:
            return self.x, self.y

    def show(self,game):
        posX = self.x * game.size_snake + game.border_size
        posY = self.y * game.size_snake + game.border_size
        pygame.draw.circle(game.pygame_display, game.food_color, (posX + game.size_snake // 2, posY + game.size_snake // 2), game.size_snake // 2)
