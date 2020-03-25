from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
import random
import numpy as np
from enum import Enum
from operator import add
import math
import json

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
        self.agent_target = 1
        self.agent_predict = 0
        self.alpha = alpha
        self.model = self.network()
        if len(weights_file)>0:
            self.model = self.network(weights_file)
        self.epsilon = epsilon
        self.actual = []
        self.memory = []
        self.old_to_food = 0
        self.new_to_food = 1

    def get_direction(self, player):
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
    
    def get_dangers(self, game, player):
        x = player.position[-1][0]
        y = player.position[-1][1]
        result = {}
        result[Direction.NORTH] = [x, y-1] in player.position or y < 1
        result[Direction.WEST] = [x + 1, y] in player.position or x + 1 >= game.cols
        result[Direction.SOUTH] = [x, y + 1] in player.position or y + 1 >= game.rows
        result[Direction.EAST] = [x - 1, y] in player.position or x < 1
        return result
    
    def get_dir_dangers(self, game, player):
        dir = self.get_direction(player)
        dangers = self.get_dangers(game, player)
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

        dangers = self.get_dir_dangers(game,game.player)

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
        return self.model.predict(state.reshape((1,12)))

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

    def network(self, weights=None):
        model = Sequential()

        model.add(Dense(units = self.neurons, input_dim = 12))
        model.add(Activation('relu'))
        model.add(Dropout(0.14))

        model.add(Dense(units = self.neurons))
        model.add(Activation('relu'))
        model.add(Dropout(0.14))

        model.add(Dense(units = self.neurons))
        model.add(Activation('relu'))
        model.add(Dropout(0.14))

        model.add(Dense(units = 3))
        model.add(Activation('softmax'))

        opt = Adam(self.alpha)
        model.compile(loss = 'mse', optimizer = opt)

        if weights:
            model.load_weights(weights)
        return model

    def replay(self):
        if len(self.memory) > self.max_memory:
            mem_part = random.sample(self.memory, self.max_memory)
        else:
            mem_part = self.memory
        for state_old, state_new, action, reward, done in mem_part:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([state_new]))[0])
            target_f = self.model.predict(np.array([state_old]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state_old]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state_old, state_new, action, reward, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(state_new.reshape((1, 12)))[0])
        target_f = self.model.predict(state_old.reshape((1, 12)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state_old.reshape((1, 12)), target_f, epochs=1, verbose=0)
        self.memory.append((state_old, state_new, action, reward, done))