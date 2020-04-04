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

class Experience(object):
    def __init__(self, state_old, state_new, action, reward, done):
        self.state_old = state_old
        self.state_new = state_new
        self.action = action
        self.reward = reward
        self.done = done

class SnakeAgent(object):

    def __init__(self, epsilon, gamma, alpha, weights_file):
        self.gamma = gamma
        self.neurons = 120
        self.max_memory = 1000
        self.alpha = alpha
        self.model = self.__get_network(weights_file)
        self.epsilon = epsilon
        self.memory = []
        self.old_to_food = 0
        self.new_to_food = 1

    def __get_direction(self, snake):
        speeds = [snake.x_speed, snake.y_speed]
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
    
    def __get_dangers(self, game, snake):
        x = snake.body[-1][0]
        y = snake.body[-1][1]
        result = {}
        result[Direction.NORTH] = [x, y-1] in snake.body or y < 1
        result[Direction.WEST] = [x + 1, y] in snake.body or x + 1 >= game.cols
        result[Direction.SOUTH] = [x, y + 1] in snake.body or y + 1 >= game.rows
        result[Direction.EAST] = [x - 1, y] in snake.body or x < 1
        return result
    
    def __get_dir_dangers(self, game, snake):
        dir = self.__get_direction(snake)
        dangers = self.__get_dangers(game, snake)
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
        a = game.snake.body[-1][0]-game.food.x
        b = game.snake.body[-1][1]-game.food.y
        self.old_to_food = self.new_to_food
        self.new_to_food = math.sqrt(a*a+b*b)

        dangers = self.__get_dir_dangers(game,game.snake)

        state = [
            dangers[0],
            dangers[1],
            dangers[2],

            game.snake.x_speed == -1, 
            game.snake.x_speed == 1,
            game.snake.y_speed == -1, 
            game.snake.y_speed == 1,
            game.food.x < game.snake.x, 
            game.food.x > game.snake.x,
            game.food.y < game.snake.y, 
            game.food.y > game.snake.y,
            self.new_to_food < self.old_to_food
            ]
        
        real_state = [1 if s else 0 for s in state]

        return np.array(real_state)
    
    def get_prediction(self, state):
        return self.model.predict(self.__shape_state(state))

    def __shape_state(self,state):
        return np.array([state])

    def get_reward(self,game):
        reward = 0
        if game.game_over:
            reward = -20
            return reward
        if game.snake.must_grow:
            reward = 10
            return reward
        if self.new_to_food < self.old_to_food:
            reward = 0.2
        elif self.new_to_food > self.old_to_food:
            reward = -0.1
        return reward

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
        result = reward
        if not done:
            result += self.gamma * np.amax(self.get_prediction(state)[0])
        return result

    def restart(self):
        mem_part = random.sample(self.memory, min(len(self.memory),self.max_memory))
        for exp in mem_part:
            self.__train(exp)

    def train_memory(self, state_old, state_new, action, reward, done):
        exp = Experience(state_old, state_new, action, reward, done)
        self.__train(exp)
        self.memory.append(exp)
    
    def __train(self, exp):
        target = self.get_prediction(exp.state_old)
        target[0][np.argmax(exp.action)] = self.__get_target_reward(exp.reward, exp.state_new, exp.done)
        self.model.fit(self.__shape_state(exp.state_old), target, epochs=1, verbose=0)
        