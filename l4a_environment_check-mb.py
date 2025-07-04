#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np 
import pandas as pd
from scipy.special import softmax 
import time 

from copy import deepcopy

from IPython.display import clear_output


import matplotlib.pyplot as plt 
import seaborn as sns 

import sys 
sys.path.append("..") 
from utils.viz import viz 
viz.get_style()


# In[32]:


class two_stage:
    nS = 9
    nA = 2
    nC = 2 
    nR = 3
    s_termination = list(range(5,9))
    
    def __init__(self,seed=873):
        '''A MDP is a 5-element-tuple

        S: state space
        A: action space
        T: transition function
        C: certainty
        R: reward condition
        '''
        self.rng = np.random.RandomState(seed)
        self.nS = two_stage.nS
        self.nA = two_stage.nA
        self.nC = two_stage.nC 
        self._init_state()
        self._init_action()
        self._init_trans_fn()
        self._init_reward()


    
    def _init_state(self):
        self.S  = np.arange(two_stage.nS)

    def _init_action(self):
        self.A  = np.arange(two_stage.nA)

    def _init_certainty(self):
        self.C  = np.arange(two_stage.nC)

    def _init_reward(self):
        self.R  = np.arange(two_stage.nR)

    def _init_trans_fn(self):
        '''T(s'|s,a)'''
        def pro(C):
            if C == 0:
                prob = 0.9
            elif C == 1:
                prob = 0.5
            return prob
        
        self.t = {
                0: {0: {0: [0, pro(0), 1-pro(0), 0, 0, 0, 0, 0, 0],
                        1: [0, pro(1), 1-pro(1), 0, 0, 0, 0, 0, 0]},
                    1: {0: [0, 0, 0, pro(0), 1-pro(0), 0, 0, 0, 0],
                        1: [0, 0, 0, pro(1), 1-pro(1), 0, 0, 0, 0]},},

                1: {0: {0: [0, 0, 0, 0, 0, 0, 1-pro(0), pro(0), 0],
                        1: [0, 0, 0, 0, 0, 0, 1-pro(1), pro(1), 0]},
                    1: {0: [0, 0, 0, 0, 0, 1-pro(0), pro(0), 0, 0],
                        1: [0, 0, 0, 0, 0, 1-pro(1), pro(1), 0, 0]},},

                2: {0: {0: [0, 0, 0, 0, 0, 0, 1-pro(0), pro(0), 0],
                        1: [0, 0, 0, 0, 0, 0, 1-pro(1), pro(1), 0]},
                    1: {0: [0, 0, 0, 0, 0, 1-pro(0), 0, pro(0), 0],
                        1: [0, 0, 0, 0, 0, 1-pro(1), 0, pro(1), 0]},},
                
                3: {0: {0: [0, 0, 0, 0, 0, 0, 0, pro(0), 1-pro(0)],
                        1: [0, 0, 0, 0, 0, 0, 0, pro(1), 1-pro(1)]},
                    1: {0: [0, 0, 0, 0, 0, 1-pro(0), 0, 0, pro(0)],
                        1: [0, 0, 0, 0, 0, 1-pro(1), 0, 0, pro(1)]},},

                4: {0: {0: [0, 0, 0, 0, 0, 1-pro(0), 0, pro(0), 0],
                        1: [0, 0, 0, 0, 0, 1-pro(1), 0, pro(1), 0]},
                    1: {0: [0, 0, 0, 0, 0, pro(0), 0, 0, 1-pro(0)],
                        1: [0, 0, 0, 0, 0, pro(1), 0, 0, 1-pro(1)]},},
                }

        
    def trans_fn(self,s,a,C):
        self.T = self.t[s][a][C]
        return self.T         
    
    def _init_reward(self):
        '''R(r|s',a)''' 
    
        self.r = {
                    0:[0,0,0,0,0,0,.1,.2,.4],
                    1:[0,0,0,0,0,0,.1,0,0],
                    2:[0,0,0,0,0,0,0,.2,0],
                }   
          
    def reward_fn(self,W,s):
        self.R = self.r[W][s]
        return self.R
    
    def reset(self):
        '''always start with state=0
        '''
        self.s = 0
        self.done = False 
        return self.s 
    
    def step(self,a,C,W):
        
        # get next state 
        p_s_next = self.trans_fn(self.s,a,C)
        s_next = self.rng.choice(self.S, p=p_s_next)
        # get the reward
        obj_r = self.reward_fn(W,s_next)
        # check the termination
        if s_next > 4: self.done = True 
        # move on 
        self.s = s_next 

        return self.s, obj_r, self.done


# In[33]:


def sim(agent_fn, params, n_episode = 5000, seed=1234): 
    # agent_fn = lossarbi specific,high; flexible,low; specific, high; flexible, low
    cols = ['uncertainty','goal','pi1','a1','s2','r1','a2','s3','r2']
    sim_data = {col: [] for col in cols}
    nS, nA= 9, 2 
    rng = np.random.RandomState(seed)
    agent = agent_fn(nS, nA, rng, params=params)
    env = two_stage()
    certainty = 0
    goal = 0
    for epi in range(n_episode):
        if epi < 2500:
            certainty = 0
        else:
            certainty = 1

        if epi // 1250 == 0 or epi // 1250 == 2:
            goal = 0
        else:
            goal = 1
            
        agent_fn.n = goal
        # stage 1
        s1 = env.reset()          # get state 
        a1, pi1 = agent.make_move(s1)           # get action
        # stage 2
        s2, r1, done = env.step(a1,C=certainty,W=goal)  # get state; C is the uncertainty, W is the reward condition
        agent.learn1(s1, a1, s2, r1)
        a2, pi2 = agent.make_move(s2)           # get action 
        s3, r2, done = env.step(a2,C=certainty,W=goal)      # get reward
        agent.learn2(s1, a1, s2, r1, a2, s3, r2)   
        # save
        sim_data['uncertainty'].append(certainty)
        sim_data['goal'].append(goal)
        sim_data['pi1'].append(pi1[0])   
        sim_data['a1'].append(a1)
        sim_data['s2'].append(s2)
        sim_data['r1'].append(r1)
        sim_data['a2'].append(a2)
        sim_data['s3'].append(s3)
        sim_data['r2'].append(r2)

    return sim_data


# ## The behavior of arbitrator who minimizes loss

# In[34]:


class lossarbi:
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
        self.alpha1 = params['alpha1']
        self.alpha2 = params['alpha2']
        self.beta1  = params['beta1']
        self.beta2  = params['beta2']
        self.lmbda  = params['lmbda']
        self.alpha_u = params['alpha_u']
        self.theta1 = params['theta1'] # transition matrix learning rate
        self.p      = params['p']
        self.eta    = params['eta']
        self.u      = 0
        self.w      = 0.5
        self.n = 0 # for model based learner, kind of reward condition
        self.reward = np.zeros([3, nS])
        self.reward[0:,5:] = [[0, .1, .2 ,.4],[0,.1,0,0],[0,0,.2,0]] # should the model based learner learn the reward through trials? 
        #or should we just suppose he knows it immediately
        self.rep_a  = np.zeros([self.nA])
  

    def make_move(self, s):
        q_mb = self.Q_mb[s, :]
   
        beta = self.beta1 if s==0 else self.beta2
        q = q_mb + self.p*self.rep_a
        pi = softmax(beta*q)
        return self.rng.choice(self.nA, p=pi), pi 
    
    def learn1(self, s1, a1, s2, r1): # r1 = 0, so no use

        #model based update
        self.P[s1, a1, s2] = self.P[s1, a1, s2] * (1 - self.theta1) + self.theta1
        if a1 == 0:
            self.P[s1, a1, 3-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.P[s1, a1, 7-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])

        
       

    def learn2(self, s1, a1, s2, r1, a2, s3, r2):
        # model-based update
        ##level 2 update
        ###first update the perception of the environment
        self.P[s2, a2, s3] = self.P[s2, a2, s3] * (1 - self.theta1) + self.theta1
        for i in range(9):
            if self.P[s2, a2, i] != 0 and i != s3:
                self.P[s2, a2, i] = 1 - self.P[s2, a2, s3].copy() ###then update the Q value for model based learner
                self.Q_mb[s2, a2] = self.P[s2, a2, s3] * self.reward[self.n, s3] + self.P[s2, a2, i] * self.reward[self.n, i]

        ##level 1 update
        if a1 == 0:
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])

        
        
        
        #update perseveration
        self.rep_a = np.eye(self.nA)[a1]

       


# In[35]:


agent_fn = lossarbi
params = {'alpha1': .4,
          'alpha2': .5, 
          'beta1': 6, 
          'beta2': 4, 
          'lmbda': .6,
          'alpha_u': .5,
          'p': .1,
          'eta':.99,
          'theta1':0.3
          }
sim_LA_data = sim(agent_fn, params)