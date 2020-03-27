import random
import numpy as np
import math
from enum import Enum
from operator import add
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Dense

class Direction(Enum):
    NORTH = 1
    WEST = 2
    SOUTH = 3
    EAST = 4

class SnakeAgent(object):

    def __init__(self, epsilon, gamma, alpha, weights_file):
        self.reward = 0
        self.gamma = gamma
        self.neurons = 120
        self.max_memory = 1000
        self.alpha = alpha
        self.model = self.__get_network(weights_file)
        self.epsilon = epsilon
        self.memory = []
        self.old_to_food = 0
        self.new_to_food = 1

    def __get_direction(self, player):
        speeds = [player.x_speed, player.y_speed]
        if speeds == [1,0]:
            return Direction.WEST
        elif speeds == [-1,0]:
            return Direction.EAST
        elif speeds == [0,1]:
            return Direction.SOUTH
        elif speeds == [0,-1]:
            return Direction.NORTH
        else:
            return 0
    
    def __get_dangers(self, game, player):
        x = player.position[-1][0]
        y = player.position[-1][1]
        result = {}
        result[Direction.NORTH] = [x, y-1] in player.position or y < 1
        result[Direction.WEST] = [x + 1, y] in player.position or x + 1 >= game.cols
        result[Direction.SOUTH] = [x, y + 1] in player.position or y + 1 >= game.rows
        result[Direction.EAST] = [x - 1, y] in player.position or x < 1
        return result
    
    def __get_dir_dangers(self, game, player):
        dir = self.__get_direction(player)
        dangers = self.__get_dangers(game, player)
        result = [True,True,True]
        if dir == Direction.NORTH:
            result = [
                dangers[Direction.NORTH],
                dangers[Direction.EAST],
                dangers[Direction.WEST]
            ]
        elif dir == Direction.WEST:
            result = [
                dangers[Direction.WEST],
                dangers[Direction.NORTH],
                dangers[Direction.SOUTH]
            ]
        elif dir == Direction.SOUTH:
            result = [
                dangers[Direction.SOUTH],
                dangers[Direction.WEST],
                dangers[Direction.EAST]
            ]
        elif dir == Direction.EAST:
            result = [
                dangers[Direction.EAST],
                dangers[Direction.SOUTH],
                dangers[Direction.NORTH]
            ]
        return result

    def get_state(self, game):
        a = game.player.position[-1][0]-game.food.x
        b = game.player.position[-1][1]-game.food.y
        self.old_to_food = self.new_to_food
        self.new_to_food = math.sqrt(a*a+b*b)

        dangers = self.__get_dir_dangers(game,game.player)

        state = [
            dangers[0],
            dangers[1],
            dangers[2],

            game.player.x_speed == -1, 
            game.player.x_speed == 1,
            game.player.y_speed == -1, 
            game.player.y_speed == 1,
            game.food.x < game.player.x, 
            game.food.x > game.player.x,
            game.food.y < game.player.y, 
            game.food.y > game.player.y,
            self.new_to_food < self.old_to_food
            ]
        
        real_state = [1 if s else 0 for s in state]

        return np.array(real_state)
    
    def get_prediction(self, state):
        return self.model.predict(self.__shape_state(state))

    def __shape_state(self,state):
        return np.array([state])

    def get_reward(self, game):
        self.reward = 0
        if game.game_over:
            self.reward = -20
            return self.reward
        if game.player.must_grow:
            self.reward = 10
            return self.reward
        if self.new_to_food < self.old_to_food:
            self.reward = 0.2
        elif self.new_to_food > self.old_to_food:
            self.reward = -0.1
        return self.reward

    def __get_network(self, weights=None):
        model = Sequential()
        # layer 1
        model.add(Dense(units = self.neurons, input_dim = 12))
        model.add(Activation('relu'))
        model.add(Dropout(0.14))
        # layer 2
        model.add(Dense(units = self.neurons))
        model.add(Activation('relu'))
        model.add(Dropout(0.14))
        # layer 3
        model.add(Dense(units = self.neurons))
        model.add(Activation('relu'))
        model.add(Dropout(0.14))
        # output
        model.add(Dense(units = 3))
        model.add(Activation('softmax'))

        model.compile(loss = 'mse', optimizer = Adam(self.alpha))

        if len(weights) > 0:
            model.load_weights(weights)
        return model
    
    def __get_target_reward(self, reward, state, done):
        if not done:
            return reward + self.gamma * np.amax(self.get_prediction(state)[0])
        else:
            return reward

    def restart(self):
        mem_part = random.sample(self.memory, min(len(self.memory),self.max_memory))
        for mem_cell in mem_part:
            (state_old, state_new, action, reward, done) = mem_cell
            self.__train(state_old, state_new, action, reward, done)

    def train_memory(self, state_old, state_new, action, reward, done):
        self.__train(state_old, state_new, action, reward, done)
        self.memory.append((state_old, state_new, action, reward, done))
    
    def __train(self, state_old, state_new, action, reward, done):
        target = self.get_prediction(state_old)
        target[0][np.argmax(action)] = self.__get_target_reward(reward, state_new, done)
        self.model.fit(self.__shape_state(state_old), target, epochs=1, verbose=0)
        