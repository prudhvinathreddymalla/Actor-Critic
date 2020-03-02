# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:17:47 2020

@author: Prudhvinath
"""

import gym
import torch

import numpy as np
from collections import namedtuple
from model import Model
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import matplotlib.pyplot as plt

SavedAction = namedtuple('SavedAction', ['log_prob' ,  'value'])

env = gym.make('CartPole-v0')

class A2CAgent():
    
    def __init__(self, env, gamma, lr):
        
        self.env = env
        self.inputDim = env.observation_space.shape[0]
        self.outputDim = env.action_space.n
        
        self.gamma = gamma
        self.lr = lr
        
        self.model = Model(self.inputDim, self.outputDim)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.epsilon = 1e-4
        
    def getAction(self, state):
        
        state = torch.FloatTensor(state)
        probs, stateValue = self.model.forward(state)
        
        #categorical distributions over the list of probabilities of actions
        catDist = Categorical(probs)
        action = catDist.sample()       #sample actions using dist
        
        self.model.savedActions.append(SavedAction(catDist.log_prob(action),stateValue))
        
        return action.item()
    
    def calLoss(self):
        
        R = 0
        savedActions = self.model.savedActions
        totalPolicyLoss = []        #list saves actor loss
        totalValueLoss = []         #list saves critic loss
        returns = []                #list saves the true values
        
        #calcualting the TD(0) returns
        for r in self.model.rewards[::-1]:
            
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        #standardising the returns
        returns = (returns - returns.mean()) / (returns.std() + self.epsilon)
        
        for (log_prob, value), R in zip(savedActions, returns):
            advantage = R - value.item()
            
            #policy loss
            totalPolicyLoss.append(-log_prob * advantage)
            totalValueLoss.append(F.mse_loss(value, torch.tensor([R])))

        self.optimizer.zero_grad()
        
        #adding up all loss to backpropagate
        totalLoss = torch.stack(totalPolicyLoss).sum() + torch.stack(totalValueLoss).sum()
        
        totalLoss.backward()
        self.optimizer.step()
        
        del self.model.rewards[:]
        del self.model.savedActions[:]
  
maxEpisodes = 2000
gamma = 0.99
lr = 1e-3
agent = A2CAgent(env, gamma, lr)
maxCounter = 10000      
rewardsList = []

window = 10

for episode in range(maxEpisodes):
    
    state = env.reset()
    epReward = 0
    
    for step in range(maxCounter):
        
        action = agent.getAction(state)    
        state, reward, done, _ = env.step(action)    
        agent.model.rewards.append(reward)    
        epReward += reward    
        if done:
            # env.render()
            break
    
    rewards_smooth = [np.mean(rewardsList[i:i+window]) if i > window 
                  else np.mean(rewardsList[0:i+1]) for i in 
                  range(len(rewardsList))]
    agent.calLoss()
    rewardsList.append(epReward)    
    
    if episode % 100 == 0:
        print(f'Episode: {episode}, Reward: {epReward}')


plt.figure(1, figsize = (15, 10))
plt.plot(rewardsList)
plt.plot(rewards_smooth)
plt.title('Rewards vs Episodes')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('./' + 'Rewards' + ".png", dpi = 450)
plt.show()
        
         
    
    
        
        
        
        
        
        

        
        



        
                         
                         
