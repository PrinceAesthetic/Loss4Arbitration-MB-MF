import numpy as np 
import pandas as pd
from scipy.special import softmax 
import time 

from copy import deepcopy

from IPython.display import clear_output

import matplotlib.pyplot as plt 
import seaborn as sns 

from utils.env import two_stage_task
from utils.viz import viz 
viz.get_style()


class two_stage_task:
    '''The two stage task

    The task reported in Daw et al., 2011 is a
    two-stage MDP. The task is written in the gym
    format. Here we will define the 4-tuple
    for this MDP (S, A, T, R)

    S: the state space, 
    A: the action space, 
    P: the transition fucntion, 
    R: the reward function,
    '''
    nS = 3 
    nA = 3

    def __init__(self, rho=.7, seed=2023):
        self.rho   = rho  # transition probability
        self.rng   = np.random.RandomState(seed)
        # define MDP 
        self._init_S()
        self._init_A()
        self._init_P()
        self._init_R()

    # -------- Define the task -------- #
        
    def _init_S(self):
        self.S = [0, 1, 2]

    def _init_A(self):
        self.A = [0, 1]

    def _init_P(self):
        '''The transition function

        The transition matrix is:

                 s0     s1      s2      
        s0-a0    0      t       1-t   
        s0-a1    0      1-t     t  

        s1-a0    1      0       0        
        s1-a1    1      0       0   
    
        s2-a0    1      0       0      
        s2-a1    1      0       0    
        '''
        self.P = np.zeros([self.nS, self.nA, self.nS])
        # state == 0 
        self.P[0, 0, :] = [0, self.rho, 1-self.rho]
        self.P[0, 1, :] = [0, 1-self.rho, self.rho]
        # state != 0 
        self.P[1:, :, 0] = 1
        # common state 
        self.common_state = 1 if self.rho > .5 else 2
        def p_s_next(s, a):
            return self.P[s, a, :].copy()
        self.p_s_next = p_s_next

    def _init_R(self):
        '''The reward function

            the probability of getting reward

                    p(r|s, a)      
            s0-a0      0              
            s0-a1      0             

            s1-a0     .9               
            s1-a1     .4             

            s2-a0     .1           
            s2-a1     .6        
        '''
        def r_fn(s, a):
            r_mat = np.zeros([3, 2])
            r_mat[1, 0] = .9
            r_mat[1, 1] = .4
            r_mat[2, 0] = .1 
            r_mat[2, 1] = .6
            p = r_mat[s, a]
            return self.rng.choice([0, 1], p=[1-p, p])
        self.r_fn = r_fn
    
    # -------- Run the task -------- #

    def reset(self):
        '''Reset the task, always start with state=0
        '''
        self.s = 0
        self.t = -1
        self.r = 0 
        self.done = False
        info = {'stage': 0}
        return self.s, self.r, self.done, info

    def render(self, ax, a=None):
        occupancy = np.zeros([3, 5])
        cmaps = [viz.GreensMap, viz.RedsMap, viz.BluesMap]
        texts = [(r'$\alpha$', r'$\beta$'), 
                 (r'$\gamma$', r'$\delta$'),
                 (r'$\epsilon$', r'$\zeta$'),]
        if not self.done:
            occupancy[1, 1] = 1
            occupancy[1, 3] = 1
            sns.heatmap(occupancy, cmap=cmaps[self.s],
                        vmin=0, vmax=1, cbar=False,
                        ax=ax)
            ax.text(1.2, 1.7, texts[self.s][0], color='k',
                fontweight='bold', fontsize=30)
            ax.text(3.2, 1.7, texts[self.s][1], color='k',
                    fontweight='bold', fontsize=30)
            stage = 1 if self.s==0 else 2
            ax.set_title(f'Stage: {stage}')
            ax.axhline(y=0, color='k',lw=5)
            ax.axhline(y=occupancy.shape[0], color='k',lw=5)
            ax.axvline(x=0, color='k',lw=5)
            ax.axvline(x=occupancy.shape[1], color='k',lw=5)
            ax.set_axis_off()
            ax.set_box_aspect(3/5)
            if a is not None:
                square = plt.Rectangle((a*2+.9, .9), 1.2, 1.2, lw=3,
                            fill=False, edgecolor='r')
                ax.add_patch(square)
        else:
            sns.heatmap(occupancy, cmap='Greys',
                            vmin=0, vmax=1, cbar=False,
                            ax=ax)
            if self.r:
                circle = plt.Circle((2.5, 1.5), 1, 
                            facecolor=viz.Yellow, edgecolor='none')
                ax.add_patch(circle)
                ax.text(2.25, 1.7, "1", color='k',
                    fontweight='bold', fontsize=35)
            ax.set_title(f'Reward: {self.r}')
            ax.axhline(y=0, color='k',lw=5)
            ax.axhline(y=occupancy.shape[0], color='k',lw=5)
            ax.axvline(x=0, color='k',lw=5)
            ax.axvline(x=occupancy.shape[1], color='k',lw=5)
            ax.set_axis_off()
            ax.set_box_aspect(3/5)

    def step(self, a):
        '''For each trial 

        Args:
            a: take the action conducted by the agent 

        Outputs:
            s_next: the next state
            rew: reward 
            info: some info for analysis 
        '''
        if a is None: a=0
        # Rt(St, At)
        self.r = self.r_fn(int(self.s), int(a))
        # St, At --> St+1 
        s_next = self.rng.choice(self.nS, p=self.p_s_next(self.s, a))
        # if the state is common 
        if s_next != 0: self.t += 1
        # if it is the end of the trial
        self.done = 1 if s_next == 0 else 0
        # info
        info = {
            'stage': 1 if s_next==0 else 0,
            'common': 'common' if int(a)+1==s_next else 'rare',
            'rewarded': 'rewarded' if self.r else 'unrewarded',
        }
        # if self.s==0: 
        #     print(a, s_next)
        #     print(int(a)+1==s_next)
        # now at the next state St
        self.s = s_next 
        return s_next, self.r, self.done, deepcopy(info)
    

class model_free:
    '''Q learning
    '''

    def __init__(self, nS, nA, rng, params):
        self.nS = nS
        self.nA = nA
        self.rng = rng
        self.Q   = np.ones([nS, nA]) / nA 
        self.alpha = params['alpha']
        self.beta  = params['beta']
        self.gamma = params['gamma']

    def make_move(self, s):
        q = self.Q[s, :]
        pi = softmax(self.beta*q)
        return self.rng.choice(self.nA, p=pi)
    
    def learn(self, traj):
        #for traj in reversed(traj):
        traj1, traj2 = traj
        s, a, r, s_next, done = traj2
        q_tar = r #+ (1-done)*self.gamma*np.max(self.Q[s_next, :])
        delta2 = q_tar - self.Q[s, a]
        self.Q[s, a] = self.Q[s, a] + self.alpha*delta2
        s, a, r, s_next, done = traj1
        q_tar = r + (1-done)*self.gamma*np.max(self.Q[s_next, :])
        delta1 = q_tar - self.Q[s, a]
        self.Q[s, a] = self.Q[s, a] + self.alpha*(delta1+.75*delta2)\



env = two_stage_task()

cols = ['a1', 's2', 'a2', 'r2', 'block_type', 'Q']
sim_data = {col: [] for col in cols}
nS, nA = 3, 2
rng = np.random.RandomState(45456)
params = {'alpha': .2, 'beta': 10, 'gamma': .9}
agent = model_free(nS, nA, rng, params=params)
episode = 1000
for epi in range(episode):
    # if epi>50: 
    #     print(1)
    # stage 1
    Q_prev = deepcopy(agent.Q[0, :])
    s1, _, _, _ = env.reset()         # get state 
    a1 = agent.make_move(s1)          # get action
    #if epi==0: prev_a1 = a1 
    # stage 2
    s2, r1, done, info = env.step(a1) # get state 
    traj1 = (s1, a1, r1, s2, done)    # get trajectory 
    a2 = agent.make_move(a1)          # get action 
    s3, r2, done, _ =env.step(a2)     # get reward
    traj2 = (s2, a2, r2, s3, done)    # get trajectory 
    agent.learn([traj1, traj2])   
    Q_after = deepcopy(agent.Q[0, :])
    #print(info['common'])
    # save
    sim_data['a1'].append(a1)
    sim_data['s2'].append(s2)
    sim_data['a2'].append(a2)
    sim_data['r2'].append(r2) 
    sim_data['block_type'].append(info['common'])
    sim_data['Q'].append(str(Q_prev - Q_after))
    # sim_data['if_stay'].append(prev_a1==a1)
    #prev_a1 = a1
sim_data = pd.DataFrame.from_dict(sim_data) 
sim_data['if_rewarded'] = sim_data['r2'].map(
    {0: 'unrewarded', 1: 'rewarded'}
)   
sim_data['a1_next'] = sim_data['a1'].shift(-1)
sim_data['if_stay'] = sim_data.apply(
    lambda x: x['a1'] == x['a1_next']
, axis=1)

print(1)