import os
import pickle
import numpy as np
import multiprocessing as mp

import fit_functions as fit
import loss4arbitration_fit as agent
from loss4arbitration_fit import two_stage

def get_pool(n_fit, n_cores=None):
    n_cores = n_cores if n_cores else int(mp.cpu_count() * 0.5)
    print(f"Using {n_cores} cores for {n_fit} fits")
    return mp.Pool(n_cores)

if __name__ == '__main__':
    # STEP 0: 并行池
    n_fits = 80
    pool = get_pool(n_fits, None)

    # STEP 1: 数据
    dir = './PKL_DATA'
    agent_name = ['larbi_leaky']  # 可同时拟合多个模型
    group_files = {"ALL": f"{dir}/ALL_SUBJECTS.pkl"}

    # STEP 2: 环境
    seed = 2025
    rng = np.random.RandomState(seed)
    task_env = two_stage()

    # STEP 3: 拟合
    os.makedirs(os.path.join(dir, 'fitdata'), exist_ok=True)

    for group_name, pkl_path in group_files.items():
        with open(pkl_path, 'rb') as f:
            agent_data = pickle.load(f)

        for name in agent_name:
            task_agent = getattr(agent, name)
            print(f'\n===== START MODEL: {name} =====')
            all_results = fit.fl(pool, task_agent, agent_data, task_env, n_fits)
            print(f'===== END   MODEL: {name} =====\n')

            

            output_path = os.path.join(dir, 'fitdata', f'fitresults_{group_name}_{name}.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"已保存 {group_name} 组 {name} 模型结果 -> {output_path}")

    pool.close()
    pool.join()
