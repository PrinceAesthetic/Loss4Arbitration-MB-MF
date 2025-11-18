import numpy as np 
import pandas as pd
from scipy.special import softmax 
from scipy.special import expit
import time 
import copy

from copy import deepcopy



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

                2: {0: {0: [0, 0, 0, 0, 0, 1-pro(0), pro(0), 0, 0],
                        1: [0, 0, 0, 0, 0, 1-pro(1), pro(1), 0, 0]},
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
                    0:[0,0,0,0,0,0,10,20,40],
                    1:[0,0,0,0,0,0,10,0,0],
                    2:[0,0,0,0,0,0,0,20,0],
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

'''
def sim(agent_fn, params, n_episode = 150, seed=873): 
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
        
        # Block 1 0-36: fixed 20, uncertainty=0.9 
        # Block 2 37-74: flexible, uncertainty=0.9  
        # Block 3 75-111: fixed 10, uncertainty=0.5 
        # Block 4 112-149: flexible, uncertainty=0.5 
        
        if epi < 37: 
            goal = 2 
            certainty = 0  
        elif epi < 75:   
            goal = 0  
            certainty = 0  
        elif epi < 112:
            goal = 1  
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
        a1 = agent.make_move(s1)           # get action
        # stage 2
        s2, r1, done = env.step(a1,C=certainty,W=goal)  # get state; C is the uncertainty, W is the reward condition
        a2 = agent.make_move(s2)           # get action 
        # stage 3
        s3, r2, done = env.step(a2,C=certainty,W=goal)      # get reward
        agent.learn(s1, a1, s2, r1, a2, s3, r2, goal)   
        # save
        sim_data['w'].append(agent.w2)
        sim_data['Q_mf'].append(agent.Q_mf[s2,a2])
        sim_data['Q_mb'].append(agent.Q_mb[s2,a2])  
        sim_data['r'].append(r2)
    
    return sim_data
'''

#######################################################

# loss for arbitration, original version

#######################################################

class lossarbi:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'lossarbi'  # fl() 打印用
    p_name = ['alpha1','alpha2','beta1','beta2','lmbda','alpha_u','p','eta','theta1','theta2']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1), (0,1),   # alpha1, alpha2
              (0,1),(0,1),  # beta1, beta2
              (0,1), (0,1),   # lmbda, alpha_u
              (0,0.5), (0.9,1),   # p, eta
              (0,1), (0,1)]   # theta1, theta2
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [lambda x: x]*len(p_name)

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
        self.theta1 = params['theta1']
        self.theta2 = params['theta2'] 
        self.p      = params['p']
        self.eta    = params['eta']
        # self.u1      = 0
        # self.w1      = 1/(1+np.exp(-self.u1)) # arbitration weight for level 1
        self.u2      = 0
        self.w2      = 1/(1+np.exp(-self.u2)) # arbitration weight for level 2
        self.reward = np.zeros([3, nS])
        self.reward[0:,5:] = [[0, 10, 20 ,40],[0,10,0,0],[0,0,20,0]] # should the model based learner learn the reward through trials? 
        #or should we just suppose he knows it immediately
        self.rep_a1  = np.zeros([self.nA])
        self.rep_a2  = np.zeros([self.nA])


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
        beta = self.beta1 if s==0 else self.beta2
        q = q_net + self.p*self.rep_a1 if s==0 else q_net + self.p*self.rep_a2
        pi = softmax(beta*q)
        return pi 
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal): # r1 = 0, so no use
        #save the Q values
        Q_his_mb = deepcopy(self.Q_mb)
        Q_his_mf = deepcopy(self.Q_mf)
         
        #update the arbitration weight
        w2 = deepcopy(self.w2)
        u2 = deepcopy(self.u2)
        grad_u = (w2*Q_his_mb[s2,a2] + (1-w2)*Q_his_mf[s2,a2] - self.eta*r2) * (Q_his_mb[s2,a2] - Q_his_mf[s2,a2]) * w2 * (1-w2)
        self.u2 = self.u2 - self.alpha_u*grad_u
        self.w2 = expit(self.u2)



        
        
    
        
        #model free update
        q_hat2 = self.Q_mf[s2, a2].copy()
        q_hat1 = self.Q_mf[s1, a1].copy()
        delta2 = r2 - q_hat2
        delta1 = q_hat2 - q_hat1
        # update 
        self.Q_mf[s2, a2] += self.alpha2*delta2
        self.Q_mf[s1, a1] += self.alpha1*(delta1 + self.lmbda*delta2)

        #model based update
        #update the transition matrix
        self.P[s2, a2, s3] = self.P[s2, a2, s3] * (1 - self.theta2) + self.theta2
        for i in range(9):
            if self.P[s2, a2, i] != 0 and i != s3:
                self.P[s2, a2, i] = 1 - self.P[s2, a2, s3].copy() ###then update the Q value for model based learner
                self.Q_mb[s2, a2] = self.P[s2, a2, s3] * self.reward[goal, s3] + self.P[s2, a2, i] * self.reward[goal, i]    

        self.P[s1, a1, s2] = self.P[s1, a1, s2] * (1 - self.theta1) + self.theta1
        if a1 == 0:
            self.P[s1, a1, 3-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.P[s1, a1, 7-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])
        
        
        

        # update perseveration
        self.rep_a1 = np.eye(self.nA)[a1]
        self.rep_a2 = np.eye(self.nA)[a2]


#######################################################

# random agent

#######################################################

class randomagent:
    # —— 元数据（fit_functions 会从“类”上读取）——
    name    = 'randomagent'
    p_name  = []          # 零参数
    pbnds   = []          # 零参数 → 无取值范围
    bnds    = []          # 零参数 → 无边界
    p_links = []          # 零参数 → 无链接函数

    def __init__(self, nS, nA, rng, params=None):
        self.nS, self.nA = nS, nA
        self.rng = rng

        # 让 nll 里的 _action_prob() 走“均匀概率”
        # softmax(beta * q)；beta=0 → softmax(0)=均匀分布
        self.Q_mf = np.zeros((nS, nA))
        self.Q_mb = np.zeros((nS, nA))
        self.w2   = 0.5          # 随便给个值，不影响 beta=0 的均匀概率
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.p     = 0.0         # 无惯性项
        self.rep_a1 = np.zeros(nA)
        self.rep_a2 = np.zeros(nA)

    def back_plan(self, goal):
        # 随机体无需规划
        return

    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal):
        # 随机体不学习
        return

    def make_move(self, s):
        # 仅为一致性，未被 nll 调用
        return np.ones(self.nA) / self.nA


#######################################################

# small changes to the original lossarbi

#######################################################

class larbi:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'larbi'  # fl() 打印用
    p_name = ['alpha1','alpha2','beta1','beta2','lmbda','alpha_u','theta1','theta2']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1), (0,1),   # alpha1, alpha2
              (0,2),(0,2),  # beta1, beta2
              (0,1), (0,1),   # lmbda, alpha_u
              (0,1), (0,1)]   # theta1, theta2
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 2/(1+np.exp(-z)),
        lambda z: 2/(1+np.exp(-z)),
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 1/(1+np.exp(-z)),
    ]

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
        self.theta1 = params['theta1']
        self.theta2 = params['theta2'] 
        
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
        beta = self.beta1 if s==0 else self.beta2
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
        self.Q_mf[s2, a2] += self.alpha2*delta2
        self.Q_mf[s1, a1] += self.alpha1*(delta1 + self.lmbda*delta2)

        #model based update
        #update the transition matrix
        self.P[s2, a2, s3] = self.P[s2, a2, s3] * (1 - self.theta2) + self.theta2
        for i in range(9):
            if self.P[s2, a2, i] != 0 and i != s3:
                self.P[s2, a2, i] = 1 - self.P[s2, a2, s3].copy() ###then update the Q value for model based learner
                self.Q_mb[s2, a2] = self.P[s2, a2, s3] * self.reward[goal, s3] + self.P[s2, a2, i] * self.reward[goal, i]    

        self.P[s1, a1, s2] = self.P[s1, a1, s2] * (1 - self.theta1) + self.theta1
        if a1 == 0:
            self.P[s1, a1, 3-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.P[s1, a1, 7-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])
        
        


#######################################################

# Model Free Agent

#######################################################


class MF:
    '''SARSA
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'MF'  # fl() 打印用
    p_name = ['alpha1','alpha2','beta1','beta2','lmbda']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1), (0,1),   # alpha1, alpha2
              (0,2),(0,2),  # beta1, beta2
              (0,1)  # lmbda
              ]
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 2/(1+np.exp(-z)),                
        lambda z: 2/(1+np.exp(-z)),       
        lambda z: 1/(1+np.exp(-z)),
    ]

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
        self.alpha1 = params['alpha1']
        self.alpha2 = params['alpha2']
        self.beta1  = params['beta1']
        self.beta2  = params['beta2']
        self.lmbda  = params['lmbda']
       
        

    # no back_plan for model free
  
    def back_plan(self, goal):
        return




    def make_move(self, s):
        q_mf = self.Q_mf[s, :]
        beta = self.beta1 if s==0 else self.beta2
        pi = softmax(beta*q_mf)
        return pi
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal): # r1 = 0, so no use
        #model free update
        q_hat2 = self.Q_mf[s2, a2].copy()
        q_hat1 = self.Q_mf[s1, a1].copy()
        delta2 = r2 - q_hat2
        delta1 = q_hat2 - q_hat1
        # update 
        self.Q_mf[s2, a2] += self.alpha2*delta2
        self.Q_mf[s1, a1] += self.alpha1*(delta1 + self.lmbda*delta2)

       


#######################################################

# Model Based Agent

#######################################################


class MB:
    name   = 'MB'  # fl() 打印用
    p_name = ['beta1','beta2','theta1','theta2']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [
              (0,2),(0,2),  # beta1, beta2
              (0,1), (0,1)]   # theta1, theta2
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 2/(1+np.exp(-z)),        
    lambda z: 2/(1+np.exp(-z)),        
    lambda z: 1/(1+np.exp(-z)),       
    lambda z: 1/(1+np.exp(-z)),
    ]

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
        self.beta1  = params['beta1']
        self.beta2  = params['beta2']
        self.theta1 = params['theta1']
        self.theta2 = params['theta2'] 
        self.reward = np.zeros([3, nS])
        self.reward[0:,5:] = [[0, 10, 20 ,40],[0,10,0,0],[0,0,20,0]]
        

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
        beta = self.beta1 if s==0 else self.beta2
        pi = softmax(beta*q_mb)
        return pi
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal): # r1 = 0, so no use
        #model based update
        #update the transition matrix
        self.P[s2, a2, s3] = self.P[s2, a2, s3] * (1 - self.theta2) + self.theta2
        for i in range(9):
            if self.P[s2, a2, i] != 0 and i != s3:
                self.P[s2, a2, i] = 1 - self.P[s2, a2, s3].copy() 
                self.Q_mb[s2, a2] = self.P[s2, a2, s3] * self.reward[goal, s3] + self.P[s2, a2, i] * self.reward[goal, i]    

        self.P[s1, a1, s2] = self.P[s1, a1, s2] * (1 - self.theta1) + self.theta1
        if a1 == 0:
            self.P[s1, a1, 3-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 3-s2] * np.max(self.Q_mb[3-s2, :])
        else:
            self.P[s1, a1, 7-s2] = 1 - self.P[s1, a1, s2]
            self.Q_mb[s1, a1] = self.P[s1, a1, s2] * np.max(self.Q_mb[s2, :]) + self.P[s1, a1, 7-s2] * np.max(self.Q_mb[7-s2, :])
    








#######################################################

# Hybrid Agent

#######################################################

class hybrid:
    '''SARSA + Model-based
        fixed weight w
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'hybrid'  # fl() 打印用
    p_name = ['alpha','beta','w','theta']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1),   # alpha
              (0,2),  # beta
              (0,1), # w
              (0,1)]   # theta
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
        lambda z: 1/(1+np.exp(-z)),
        lambda z: 2/(1+np.exp(-z)),                
        lambda z: 1/(1+np.exp(-z)),       
        lambda z: 1/(1+np.exp(-z)),
    ]

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
        self.beta  = params['beta']
        self.w2     = params['w']
        self.theta = params['theta']
        

       
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



    def make_move(self, s):
        q_mf = self.Q_mf[s, :]
        q_mb = self.Q_mb[s, :]
        q_net = self.w2*q_mb + (1-self.w2)*q_mf
        beta = self.beta
        pi = softmax(beta*q_net)
        return pi
    
    def learn(self, s1, a1, s2, r1, a2, s3, r2, goal):
               
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
        
        



#######################################################

# simplified lossarbi

#######################################################

class larbi_mini:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'larbi_mini'  # fl() 打印用
    p_name = ['alpha','beta','alpha_u','theta']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1),   # alpha
              (0,2),  # beta
              (0,1),   # alpha_u
              (0,1)]   # theta
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 1/(1+np.exp(-z)),        # alpha   ∈ (0,1)   —— sigmoid
    lambda z: 2/(1+np.exp(-z)),        # beta    ∈ (0,2)   —— 2*sigmoid
    lambda z: 1/(1+np.exp(-z)),        # alpha_u ∈ (0,1)
    lambda z: 1/(1+np.exp(-z)),        # theta   ∈ (0,1)
    ]

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
        
        





#######################################################

# larbi_mini with merged learning rate

#######################################################

class larbi_star:
    '''SARSA + Model-based
    '''
    
    name   = 'larbi_star'
    p_name = ['theta','beta','alpha_u']

    
    pbnds  = [(0,1),   # theta
              (0,2),  # beta
              (0,1)   # alpha_u
              ]  
    bnds   = pbnds            

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 1/(1+np.exp(-z)),        # theta   ∈ (0,1)   —— sigmoid
    lambda z: 2/(1+np.exp(-z)),        # beta    ∈ (0,2)   —— 2*sigmoid
    lambda z: 1/(1+np.exp(-z)),        # alpha_u ∈ (0,1)   —— sigmoid
    ]

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
        self.Q_mf[s2, a2] += self.theta*delta2
        self.Q_mf[s1, a1] += self.theta*(delta1 + delta2)

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
        
        










#######################################################

# larbi_mini, two layers of arbitration
# currently wrong, because the agent does not have access to the ground truth of the first stage

#######################################################

class larbi_2:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'larbi_2'  # fl() 打印用
    p_name = ['alpha','beta','alpha_u1','alpha_u2','theta']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1),   # alpha
              (0,2),  # beta
              (0,1),   # alpha_u1
              (0,1),   # alpha_u2
              (0,1)]   # theta
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 1/(1+np.exp(-z)),        # alpha   ∈ (0,1)   —— sigmoid
    lambda z: 2/(1+np.exp(-z)),        # beta    ∈ (0,2)   —— 2*sigmoid
    lambda z: 1/(1+np.exp(-z)),        # alpha_u1 ∈ (0,1)
    lambda z: 1/(1+np.exp(-z)),        # alpha_u2 ∈ (0,1)
    lambda z: 1/(1+np.exp(-z)),        # theta   ∈ (0,1)
    ]

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
        self.alpha_u1 = params['alpha_u1']
        self.alpha_u2 = params['alpha_u2']
        self.theta = params['theta']

        self.u1      = 0
        self.w1      = 1/(1+np.exp(-self.u1)) # arbitration weight for level 1
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
            self.u1 = np.log(4)
        else:
            self.u2 = -np.log(4)
            self.u1 = -np.log(4)

        self.w2 = 1/(1+np.exp(-self.u2))
        self.w1 = 1/(1+np.exp(-self.u1))




    def make_move(self, s):
        q_mf = self.Q_mf[s, :]
        q_mb = self.Q_mb[s, :]
        q_net = 0
        if s == 0:
            q_net = self.w1*q_mb + (1-self.w1)*q_mf
        else:
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
        self.u2 = self.u2 - self.alpha_u2*grad_u
        self.w2 = expit(self.u2)

        w1 = deepcopy(self.w1)
        u1 = deepcopy(self.u1)
        grad_u1 = (w1*Q_his_mb[s1,a1] + (1-w1)*Q_his_mf[s1,a1] - r2) * (Q_his_mb[s1,a1] - Q_his_mf[s1,a1]) * w1 * (1-w1)
        self.u1 = self.u1 - self.alpha_u1*grad_u1
        self.w1 = expit(self.u1)



        
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
        
        



#######################################################

# larbi_mini improved, with decay on model based control

#######################################################

class larbi_decay:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'larbi_decay'  # fl() 打印用
    p_name = ['alpha','beta','alpha_u','theta','penalty','tau']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1),   # alpha
              (0,2),  # beta
              (0,1),   # alpha_u
              (0,1),   # theta
              (0,10),   # penalty
              (0,10),   # tau
              ]   
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 1/(1+np.exp(-z)),        # alpha   ∈ (0,1)   —— sigmoid
    lambda z: 2/(1+np.exp(-z)),        # beta    ∈ (0,2)   —— 2*sigmoid
    lambda z: 1/(1+np.exp(-z)),        # alpha_u ∈ (0,1)
    lambda z: 1/(1+np.exp(-z)),        # theta   ∈ (0,1)
    lambda z: 10/(1+np.exp(-z)),        # penalty ∈ (0,10)
    lambda z: 10/(1+np.exp(-z)),        # tau ∈ (0,10)
    ]

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
        self.penalty = params['penalty']
        self.tau = params['tau']
        

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
        '''
        if goal != 0:
            self.u2 = np.log(4)
        else:
            self.u2 = -np.log(4)

        self.w2 = 1/(1+np.exp(-self.u2))
        '''




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
        
        # similarity gate
        tau = max(self.tau, 1e-6)
        sim = np.exp(-np.abs(Q_his_mb[s2,a2] - Q_his_mf[s2,a2]) / tau)

        #update the arbitration weight
        w2 = deepcopy(self.w2)
        u2 = deepcopy(self.u2)
        pred_grad_u = (w2*Q_his_mb[s2,a2] + (1-w2)*Q_his_mf[s2,a2] - r2) * (Q_his_mb[s2,a2] - Q_his_mf[s2,a2]) * w2 * (1 - w2)
        pen_grad_u = self.penalty * sim * w2 * (1 - w2)
        grad_u = pred_grad_u + pen_grad_u
        self.u2 = u2 - self.alpha_u*grad_u
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
        









#######################################################

# larbi_mini improved, with leaky arbitration weight

#######################################################

class larbi_leaky:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'larbi_leaky'  # fl() 打印用
    p_name = ['alpha','beta','alpha_u','theta','leak','tau']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1),   # alpha
              (0,2),  # beta
              (0,1),   # alpha_u
              (0,1),   # theta
              (0,10),   # leak
              (0,10),   # tau
              ]   
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 1/(1+np.exp(-z)),        # alpha   ∈ (0,1)   —— sigmoid
    lambda z: 2/(1+np.exp(-z)),        # beta    ∈ (0,2)   —— 2*sigmoid
    lambda z: 1/(1+np.exp(-z)),        # alpha_u ∈ (0,1)
    lambda z: 1/(1+np.exp(-z)),        # theta   ∈ (0,1)
    lambda z: 10/(1+np.exp(-z)),        # leak ∈ (0,10)
    lambda z: 10/(1+np.exp(-z)),        # tau ∈ (0,10)
    ]

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
        self.leak = params['leak']
        self.tau = params['tau']
        

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
        '''
        if goal != 0:
            self.u2 = np.log(4)
        else:
            self.u2 = -np.log(4)

        self.w2 = 1/(1+np.exp(-self.u2))
        '''




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
        
        # similarity gate
        tau = max(self.tau, 1e-6)
        sim = np.exp(-np.abs(Q_his_mb[s2,a2] - Q_his_mf[s2,a2]) / tau)

        #update the arbitration weight
        w2 = deepcopy(self.w2)
        u2 = deepcopy(self.u2)
        pred_grad_u = (w2*Q_his_mb[s2,a2] + (1-w2)*Q_his_mf[s2,a2] - r2) * (Q_his_mb[s2,a2] - Q_his_mf[s2,a2]) * w2 * (1 - w2)
        leaky_grad_u = self.leak * sim * u2
        grad_u = pred_grad_u + leaky_grad_u
        self.u2 = u2 - self.alpha_u*grad_u
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





#######################################################

# larbi_mini, but second stage Q_MB = Q_MF? 
# but then what about goal changes?

#######################################################

class larbi_mini:
    '''SARSA + Model-based
    '''
    # ↓↓↓ 这些是“类属性”，fd()/fl() 会直接从类上读取 ↓↓↓
    name   = 'larbi_mini'  # fl() 打印用
    p_name = ['alpha','beta','alpha_u','theta']

    # 参数边界（自己可以按需要再细调）
    pbnds  = [(0,1),   # alpha
              (0,2),  # beta
              (0,1),   # alpha_u
              (0,1)]   # theta
    bnds   = pbnds            # 若暂不做变换，优化的 bounds 用同一套

    # 链接函数（若没做 reparam 就用恒等）
    p_links = [
    lambda z: 1/(1+np.exp(-z)),        # alpha   ∈ (0,1)   —— sigmoid
    lambda z: 2/(1+np.exp(-z)),        # beta    ∈ (0,2)   —— 2*sigmoid
    lambda z: 1/(1+np.exp(-z)),        # alpha_u ∈ (0,1)
    lambda z: 1/(1+np.exp(-z)),        # theta   ∈ (0,1)
    ]

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
        
        