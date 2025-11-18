#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np 
import pandas as pd
from scipy.special import softmax 
from scipy.special import expit
import time 
import copy

from copy import deepcopy
from loss4arbitration_fit import two_stage


# In[17]:


def sim(agent_fn, params, n_episode = 150, seed=885): 
    # agent_fn = lossarbi specific,high; flexible,low; specific, high; flexible, low
    cols = ['w','Q_mf','Q_mb','r']
    sim_data = {col: [] for col in cols}
    nS, nA= 9, 2 
    rng = np.random.RandomState(seed)
    agent = agent_fn(nS, nA, rng, params=params)
    env = two_stage()
    
    prev_goal = None
    
    certainty = 0
    goal = 2
    for epi in range(n_episode):
        
        if epi < 37: 
            goal = 0 
            certainty = 0  
        elif epi < 75:   
            goal = 0  
            certainty = 0  
        elif epi < 112:
            goal = 0  
            certainty = 1  
        else:  
            goal = 0  
            certainty = 1 
        
        
        # recompute MB values when task block changes(BACK PLANNING)
        if goal != prev_goal:
            agent.back_plan(goal)
            prev_goal = goal
            
        # stage 1
        s1 = env.reset()          # get state 
        a1 = rng.choice(nA, p=agent.make_move(s1))           # get action
        # stage 2
        s2, r1, done = env.step(a1,C=certainty,W=goal)  # get state; C is the uncertainty, W is the reward condition
        a2 = rng.choice(nA, p=agent.make_move(s2))           # get action 
        # stage 3
        s3, r2, done = env.step(a2,C=certainty,W=goal)      # get reward
        agent.learn(s1, a1, s2, r1, a2, s3, r2, goal)   
        # save
        sim_data['w'].append(agent.w2)
    
    return sim_data


# In[18]:


class larbi_mini:
    '''SARSA + Model-based
    '''
    

    def __init__(self, nS, nA, rng, params):
        self.nS = nS
        self.nA = nA
        self.rng = rng
        self.Q_mf = np.zeros([nS, nA]) 
        self.Q_mb = np.zeros([nS, nA])
        '''
        Agent's perception of the environment, 
        in which he starts with equal probabilities 
        and gradually learn the approximate transition matrix
        '''  
        self.P    = np.zeros([nS, nA, nS])
        self.P[0,0,[1,2]] = .5
        self.P[0,1,[3,4]] = .5
        self.P[1,0,[6,7]] = .5
        self.P[1,1,[5,6]] = .5
        self.P[2,0,[5,6]] = .5
        self.P[2,1,[5,7]] = .5
        self.P[3,0,[7,8]] = .5
        self.P[3,1,[5,8]] = .5
        self.P[4,0,[5,7]] = .5
        self.P[4,1,[5,8]] = .5
        self.P[5:,:,0] = 1
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.alpha_u = params['alpha_u']
        self.theta = params['theta']

        # self.u1      = 0
        # self.w1      = 1/(1+np.exp(-self.u1)) # arbitration weight for level 1
        self.u2      = 0
        self.w2      = 1/(1+np.exp(-self.u2)) # arbitration weight for level 2
        self.reward = np.zeros([3, nS])
        self.reward[0:,5:] = [[0, 10, 20 ,40],[0,10,0,0],[0,0,20,0]]
        

    # back_plan involves back planning and arbitration weight initialization. 
    # When the goal is flexible, initialize with w = 0.2,
    # When the goal is fixed, initialize with w = 0.8.

    def back_plan(self, goal):
        """
        Recompute model-based Q for all states given certainty and goal.
        """
        
        term = [5,6,7,8]
        Q_backplan = np.zeros_like(self.Q_mb)
       
        for s in range(1, 5):
            for a in range(self.nA):
                Q_backplan[s, a] = self.P[s, a, term] @ self.reward[goal, term]
        
        v2 = np.max(Q_backplan[1:5, :], axis=1)  # shape (4,) for s=1..4
        for a in range(self.nA):
            Q_backplan[0, a] = self.P[0, a, 1:5] @ v2

        self.Q_mb[:] = Q_backplan

        if goal != 0:
            self.u2 = np.log(4)
        else:
            self.u2 = -np.log(4)

        self.w2 = 1/(1+np.exp(-self.u2))




    def make_move(self, s):
        q_mf = self.Q_mf[s, :]
        q_mb = self.Q_mb[s, :]
        q_net = self.w2*q_mb + (1-self.w2)*q_mf
        beta = self.beta
        pi = softmax(beta*q_net)
        return pi
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal): # r1 = 0, so no use
        #save the Q values
        Q_his_mb = deepcopy(self.Q_mb)
        Q_his_mf = deepcopy(self.Q_mf)
         
        #update the arbitration weight
        w2 = deepcopy(self.w2)
        u2 = deepcopy(self.u2)
        grad_u = (w2*Q_his_mb[s2,a2] + (1-w2)*Q_his_mf[s2,a2] - r2) * (Q_his_mb[s2,a2] - Q_his_mf[s2,a2]) * w2 * (1-w2)
        self.u2 = self.u2 - self.alpha_u*grad_u
        self.w2 = expit(self.u2)



        
        #model free update
        q_hat2 = self.Q_mf[s2, a2].copy()
        q_hat1 = self.Q_mf[s1, a1].copy()
        delta2 = r2 - q_hat2
        delta1 = q_hat2 - q_hat1
        # update 
        self.Q_mf[s2, a2] += self.alpha*delta2
        self.Q_mf[s1, a1] += self.alpha*(delta1 + delta2)

        #model based update
        #update the transition matrix
        self.P[s2, a2, s3] = self.P[s2, a2, s3] * (1 - self.theta) + self.theta
        for i in range(9):
            if self.P[s2, a2, i] != 0 and i != s3:
                self.P[s2, a2, i] = 1 - self.P[s2, a2, s3].copy() ###then update the Q value for model based learner
                self.Q_mb[s2, a2] = self.P[s2, a2, s3] * self.reward[goal, s3] + self.P[s2, a2, i] * self.reward[goal, i]    

        self.P[s1, a1, s2] = self.P[s1, a1, s2] * (1 - self.theta) + self.theta
        if a1 == 0:
            self.P[s1, a1, 3-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.P[s1, a1, 7-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])
        


# In[19]:


class MF:
    '''SARSA
    '''

    def __init__(self, nS, nA, rng, params):
        self.nS = nS
        self.nA = nA
        self.rng = rng
        self.Q_mf = np.zeros([nS, nA]) 
        '''
        Agent's perception of the environment, 
        in which he starts with equal probabilities 
        and gradually learn the approximate transition matrix
        '''  
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.w2 = 0
        


    # no back_plan for model free
  
    def back_plan(self, goal):
        return




    def make_move(self, s):
        q_mf = self.Q_mf[s, :]
        pi = softmax(self.beta*q_mf)
        return pi
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal): # r1 = 0, so no use
        #model free update
        q_hat2 = self.Q_mf[s2, a2].copy()
        q_hat1 = self.Q_mf[s1, a1].copy()
        delta2 = r2 - q_hat2
        delta1 = q_hat2 - q_hat1
        # update 
        self.Q_mf[s2, a2] += self.alpha*delta2
        self.Q_mf[s1, a1] += self.alpha*(delta1 + delta2)


# In[20]:


class MB:
   

    def __init__(self, nS, nA, rng, params):
        self.nS = nS
        self.nA = nA
        self.rng = rng
        self.Q_mb = np.zeros([nS, nA])
        '''
        Agent's perception of the environment, 
        in which he starts with equal probabilities 
        and gradually learn the approximate transition matrix
        '''  
        self.P    = np.zeros([nS, nA, nS])
        self.P[0,0,[1,2]] = .5
        self.P[0,1,[3,4]] = .5
        self.P[1,0,[6,7]] = .5
        self.P[1,1,[5,6]] = .5
        self.P[2,0,[5,6]] = .5
        self.P[2,1,[5,7]] = .5
        self.P[3,0,[7,8]] = .5
        self.P[3,1,[5,8]] = .5
        self.P[4,0,[5,7]] = .5
        self.P[4,1,[5,8]] = .5
        self.P[5:,:,0] = 1
        self.beta  = params['beta']
        self.theta = params['theta']
        self.reward = np.zeros([3, nS])
        self.reward[0:,5:] = [[0, 10, 20 ,40],[0,10,0,0],[0,0,20,0]]
        self.w2 = 0
        

    def back_plan(self, goal):
        """
        Recompute model-based Q for all states given certainty and goal.
        """
        
        term = [5,6,7,8]
        Q_backplan = np.zeros_like(self.Q_mb)
       
        for s in range(1, 5):
            for a in range(self.nA):
                Q_backplan[s, a] = self.P[s, a, term] @ self.reward[goal, term]
        
        v2 = np.max(Q_backplan[1:5, :], axis=1)  # shape (4,) for s=1..4
        for a in range(self.nA):
            Q_backplan[0, a] = self.P[0, a, 1:5] @ v2

        self.Q_mb[:] = Q_backplan




    def make_move(self, s):
        q_mb = self.Q_mb[s, :]
        pi = softmax(self.beta*q_mb)
        return pi
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal): # r1 = 0, so no use
        #model based update
        #update the transition matrix
        self.P[s2, a2, s3] = self.P[s2, a2, s3] * (1 - self.theta) + self.theta
        for i in range(9):
            if self.P[s2, a2, i] != 0 and i != s3:
                self.P[s2, a2, i] = 1 - self.P[s2, a2, s3].copy() 
                self.Q_mb[s2, a2] = self.P[s2, a2, s3] * self.reward[goal, s3] + self.P[s2, a2, i] * self.reward[goal, i]    

        self.P[s1, a1, s2] = self.P[s1, a1, s2] * (1 - self.theta) + self.theta
        if a1 == 0:
            self.P[s1, a1, 3-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.P[s1, a1, 7-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])
    


# In[21]:


params_larbi_mini = {'alpha': .3,
          'beta': .01,
          'alpha_u': .005,
          'theta':0.05
          }
sim_LA_data = sim(larbi_mini, params_larbi_mini)


# In[22]:


params_MF = {'alpha': .3,
          'beta': .01
          }
sim_MF_data = sim(MF, params_MF)


# In[23]:


params_MB = {'beta': .01,
          'theta':0.05
          }
sim_MB_data = sim(MB, params_MB)


# In[24]:


# Plot arbitration weight w across episodes
import numpy as np
import matplotlib.pyplot as plt

w = np.array(sim_LA_data['w'])
episodes = np.arange(len(w))

plt.figure(figsize=(10, 4))
plt.plot(episodes, w, color='C0', label='w (arbitration weight)')
plt.ylim(0, 1)
plt.xlabel('Episode')
plt.ylabel('w')
plt.title('Arbitration weight over episodes')

# Block boundaries based on simulation design
boundaries = [37, 75, 112]
for x in boundaries:
    plt.axvline(x, color='k', linestyle='--', alpha=0.3)

# Annotate blocks: fixed/flexible as set by `goal`
blocks = [
    (0, 37, 'flexible (0.9)'),
    (37, 75, 'flexible (0.9)'),
    (75, 112, 'flexible (0.1)'),
    (112, len(w), 'flexible (0.1)')
]
for a, b, label in blocks:
    x = (a + b) / 2
    plt.text(x, 0.95, label, ha='center', va='top', fontsize=9, alpha=0.8)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

