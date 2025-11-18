import warnings
import datetime 
import numpy as np 
from scipy.stats import gamma, norm 
from scipy.optimize import minimize
from scipy.special import softmax, psi, gammaln
from scipy.special import expit, logit


eps_ = 1e-13
max_ = 1e+13


# ===== fitdata =====

# 通用映射：z ∈ R  ↔  x ∈ (lo, hi)
def _link(z, lo, hi):
    return lo + (hi - lo) * expit(z)      # sigmoid

def _inv_link(x, lo, hi, eps=1e-8):
    x = float(np.clip(x, lo + eps, hi - eps))       # 避免端点带来 ±inf
    y = (x - lo) / (hi - lo)
    y = np.clip(y, eps, 1 - eps)
    return logit(y)

def fd(agent, data, env, method='mle',
       alg='BFGS', init=False, seed=2023,
       verbose=False):
    """
    BFGS-only 版本：在 z 空间优化；通过 sigmoid 把 z 映到 pbnds 指定的物理区间。
    """

    p_name = agent.p_name
    pbnds  = getattr(agent, 'pbnds', getattr(agent, 'bnds', []))  # 优先用 pbnds
    n_params = len(pbnds)
    n_rows   = int(getattr(data, 'shape', [0])[0]) if hasattr(data,'shape') else 0

    # 0) 零参数模型（如 random）
    if n_params == 0:
        negll = nll([], agent, env, data)
        return {
            'log_post':   -float(negll),
            'negloglike': float(negll),
            'param':      np.asarray([], dtype=float),
            'param_name': p_name,
            'n_param':    0,
            'aic':        2*0 + 2*negll,
            'bic':        0*np.log(max(n_rows,1)) + 2*negll,
        }

    # 1) 初值（物理区间）
    if init is not False:
        if isinstance(init, dict):
            param0_phys = [float(init[k]) for k in p_name]
        else:
            param0_phys = [float(x) for x in init]
    else:
        rng = np.random.RandomState(seed)
        param0_phys = [lo + (hi - lo) * rng.rand() for (lo, hi) in pbnds]

    if verbose:
        print('init (physical) =', param0_phys)

    # 2) 物理 -> z 初值
    z0 = np.array([_inv_link(x, lo, hi) for x, (lo, hi) in zip(param0_phys, pbnds)], dtype=float)

    # 3) 目标：z -> 物理（逐维 sigmoid 到各自区间） -> nll
    def obj(z):
        theta = np.array([_link(zi, lo, hi) for zi, (lo, hi) in zip(z, pbnds)], dtype=float)
        return nll(theta, agent, env, data)

    # 4) 只用 BFGS（无边界）
    res = minimize(obj, z0, method='BFGS', options={'disp': verbose})

    # 5) 还原到物理参数并打包
    theta_hat = np.array([_link(zi, lo, hi) for zi, (lo, hi) in zip(res.x, pbnds)], dtype=float)
    negll = float(nll(theta_hat, agent, env, data))

    if verbose:
        print(f"  Fitted params: {theta_hat}, Loss: {negll:.6f}")

    return {
        'log_post':   -negll,
        'negloglike': negll,
        'param':      theta_hat,
        'param_name': p_name,
        'n_param':    n_params,
        'aic':        2*n_params + 2*negll,
        'bic':        n_params*np.log(max(n_rows,1)) + 2*negll,
    }




# ===== fit parallel =====
def fp(pool, agent, data, env, n_fits=80,
       method='mle', alg='BFGS',
       init=False, seed=2025, verbose=False):

    results = [pool.apply_async(
        fd,
        args=(agent, data, env, method, alg, init, seed + 2*i, verbose)
    ) for i in range(n_fits)]

    opt_val = np.inf
    for p in results:
        res = p.get()
        if -res['log_post'] < opt_val:
            opt_val = -res['log_post']
            opt_res = res
    return opt_res



# ===== fit loop =====
def fl(pool, agent, datalist, env, n_fits):
    start_time = datetime.datetime.now()
    sub_start  = start_time
    results = []
    i = 1
    for dataname in datalist.keys():
        data = datalist[dataname]
        print(f'Fitting {agent.name} subj {dataname}, progress: {(i*100)/len(datalist):.2f}%')
        result = fp(pool, agent, data, env, n_fits,
                    method='mle', alg='BFGS',
                    init=False, seed=2025, verbose=False)
        results.append(result)
        i += 1
        sub_end = datetime.datetime.now()
        print(f'\tLOSS:{-result["log_post"]:.4f}, using {(sub_end - sub_start).total_seconds():.2f} seconds')
        sub_start = sub_end

    end_time = datetime.datetime.now()
    print('\nparallel computing spend {:.2f} seconds'.format(
          (end_time - start_time).total_seconds()))
    return results





"negloglikelihood"
# --- 放到 fit_xyf.py 里，替换原来的 nll() ---

def _action_prob(agent_obj, s, a):
    pi = agent_obj.make_move(s)
    # 防溢出保护
    return float(max(pi[a], 1e-16))


def nll(params, agent_class, env, data):
    """
    负对数似然
    - params: 优化器传入的参数向量(需转成字典喂给 agent)
    - agent_class: 例如 lossarbi
    - env: two_stage() 的实例(提供 nS/nA/终止态集合等)
    - data: DataFrame,含列 g,p,s0,a1,s1,a2,s2,r2
    """
    # 1) 把 params 向量转成字典（按 agent.p_name 顺序）; 若你还没做 p_name，可直接在这里写死映射
    
    pnames = agent_class.p_name
    theta  = {k: v for k, v in zip(pnames, params)}
    

    # 2) 构建 agent 实例（注意你的构造是 nS, nA, rng, params）
    rng = np.random.RandomState(0)
    subj = agent_class(env.nS, env.nA, rng, params=theta)

    # 3) 遍历 trial，累加 -log π(a|s)
    nLL = 0.0
    prev_g = None

    for _, row in data.iterrows():
        g  = int(row['g'])
        s0 = int(row['s0'])
        a1 = int(row['a1'])
        s1 = int(row['s1'])
        a2 = int(row['a2'])
        s_term = int(row['s2'])      # 终止态（原 sim 里的 s3）
        r2 = float(row['r2'])
        # 可用但当前没直接影响概率的列：p=row['p']; 第一阶段奖励 r1=0

        # 若目标变了，先重算 MB Q（与你的 sim/back_plan 逻辑一致）
        if g != prev_g:
            subj.back_plan(g)
            prev_g = g

        # 概率（在更新之前取）
        pi_a1 = _action_prob(subj, s0, a1)   # 第一阶段
        pi_a2 = _action_prob(subj, s1, a2)   # 第二阶段

        # 累计负对数似然
        nLL -= (np.log(pi_a1) + np.log(pi_a2))

        # 学习（一次 trial 完整经历后更新）
        subj.learn(s1=s0, a1=a1, s2=s1, r1=0.0, a2=a2, s3=s_term, r2=r2, goal=g)

    return nLL



