import torch
import random
import numpy as np
from game import SnakeGameAI,Direction,Point
from collections import deque
from model import Linear_QNet, Q_Trainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.game_no = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #calling the popleft()
        self.model = Linear_QNet(11,256,3)
        self.trainer = Q_Trainer(model=self.model , lr = LR , gamma=self.gamma)

    def get_state(self,game):
        head = game.snake[0]
        point_l = Point(head.x-20,head.y)
        point_r = Point(head.x + 20 , head.y)
        point_u = Point(head.x , head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #defining that danger is straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)) ,

            #defining that danger is right
            (dir_u and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)) ,

            #defining that danger is left
            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)) ,

            # move directions
            dir_l , dir_r , dir_u , dir_d,

            #food directions
            game.food.x < game.head.x , # food to left of snake
            game.food.x > game.head.x , # food to the right of snake
            game.food.y < game.head.y , # food to the up of snake, since -ve y-axis is on top half and +ve y is in bottom in pygame
            game.food.y > game.head.y #food to the down of the snake
        ]
        return np.array(state, dtype = int) # converting bool in state list to 0 or 1

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if max-memory reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE) #return list of tuples
        else:
            mini_sample = self.memory
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves: tradeoff btwn exploration / exploitation
        self.epsilon = 80 - self.game_no
        final_move = [0,0,0]
        if(random.randint(0,200))  < self.epsilon:
            move = random.randint(0,2) # 0 or 1 or 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # .item() to convert a tensor to int
            final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    avg_scores = []
    record_score = 0
    total_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        #perform move and get new state
        reward , done , score = game.play_step(final_move)
        new_state = agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old,final_move,reward,new_state,done)

        #remember
        agent.remember(state_old,final_move,reward,new_state,done)

        if done:
            #training long memory
            game.reset()
            agent.game_no += 1
            agent.train_long_memory()

            if score > record_score:
                record_score = score
                agent.model.save()

            print(f"GameNo: {agent.game_no} , Score: {score} , Record: {record_score}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.game_no
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)

if __name__ == "__main__":
    train()