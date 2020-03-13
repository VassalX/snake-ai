import pygame
import numpy as np
from DQN import SnakeAgent
import matplotlib.pyplot as plt
import random
import json
from scipy.stats import linregress
from snakeGame import Game

border_color = pygame.Color('SALMON')
snake_color = pygame.Color('LIGHTSEAGREEN')
food_color = pygame.Color('FIREBRICK')
background_color = pygame.Color('white')

field_rows = 20
field_cols = 20
size_snake = 20
border_size = 5
speed = 1
max_steps = 500
weights_file = ""
save_file = "weights.hdf5"
games_number = 10
epsilon = 0.8
delta_epsilon = 0.01
train = True

with open('config.json') as json_file:
    data = json.load(json_file)
    field_rows = data["field_rows"]
    field_cols = data["field_cols"]
    size_snake = data["size_snake"]
    border_size = data["border_size"]
    speed = data["speed"]
    max_steps = data["max_steps"]
    weights_file = data["weights_file"]
    games_number = data["games_number"]
    save_file = data["save_file"]
    epsilon = data["epsilon"]
    delta_epsilon = data["delta_epsilon"]
    train = data["train"]

def show_result(array_num, array_score):
    x = np.asarray(array_num)
    y = np.asarray(array_score)
    (slope, intercept, _, _, _) = linregress(x, y)
    y_pred = intercept + slope * x
    plt.plot(x, y_pred,color="green",label="Prediction")
    plt.scatter(x, y, label="Scores")
    plt.title("Result")
    plt.xlabel("game")
    plt.ylabel("score")
    plt.legend(loc='best')
    plt.show()

def play_train_game(agent):
    game = Game(field_cols, field_rows, size_snake, border_size,
        border_color, background_color, food_color, snake_color,
        speed, max_steps)

    game.make_train_step(agent,[1, 0, 0])
    agent.replay_new(agent.memory)

    while not game.game_over:
        game.make_train_step(agent)
    return game.score

def play_game(agent):
    game = Game(field_cols, field_rows, size_snake, border_size,
        border_color, background_color, food_color, snake_color,
        speed, max_steps)
    
    while not game.game_over:
        game.make_step(agent)
    return game.score

pygame.init()
agent = SnakeAgent(epsilon, weights_file)
array_score = []
array_num = []
for game_num in range(games_number):
    if train:
        score = play_train_game(agent)
        agent.epsilon -= delta_epsilon
    else:
        score = play_game(agent)
    game_num += 1
    print('#', game_num, '\tScore -', score)
    array_score.append(score)
    array_num.append(game_num)
show_result(array_num, array_score)
if train:
    agent.model.save_weights(save_file)
