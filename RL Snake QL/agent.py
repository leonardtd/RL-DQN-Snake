#################
#https://www.youtube.com/watch?v=PJl4iabBEz0
#################

from numpy.core.fromnumeric import size
import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

#Test with params
MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.01

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #control randomness
        self.gamma = 0.9 #discount rate: BELLMAN'S EQUATION
        self.memory = deque(maxlen=MAX_MEM) #popleft() if full
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #TODO: model, trainer
        
    """
    STORE 11 VALUES: [danger straight, danger right, danger left,
                    dir left, dir right, dir up, dir down,
                    food l, food r, food u, food d]
    """
    def get_state(self, game):
        head = game.snake[0] #first item in list is head

        #Check boundaries
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x , head.y - 20)
        point_d = Point(head.x , head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #11 values
        state = [
            #Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, #Food is left of head
            game.food.x > game.head.x, #Food is right
            game.food.y < game.head.y, #Food is up
            game.food.y > game.head.y #Food is down

        ]

        return np.array(state,dtype=int)


    def get_action(self,state):
        #Random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.n_games #play arround with this
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0) #Executes FORWARD method in NN
            move = torch.argmax(prediction).item() #converts to int
            final_move[move] = 1
        
        return final_move


    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) #popleft if exceeds MAX MEM
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) #List of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
    

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        #get current state
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (for 1 step)
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember in deque
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            #train the long memory, REPLAY MEMORY: EXPERIENCED REPLAY, plot the result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            #saves new highscore
            if score > record:
                record = score
                agent.model.save()

            print('Game ', agent.n_games, 'Score ', score, 'Record ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()


