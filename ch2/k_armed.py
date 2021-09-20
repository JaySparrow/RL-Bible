import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class KArmedEnv:
    r""" create num_env independent k-armed bandit problem
    stationary q_star values for each environment
    """
    def __init__(self, k: int=10, num_env: int=2000):
        self.k = k
        self.num_env = num_env
        # action space
        self.action_space = np.arange(0, k)
        # q values
        self._q_star = self._init_q_star(num_env, k) # num_env indepedent k-armed environments

    def _init_q_star(self, num_env: int, k: int):
        return np.random.normal(0, 1, (num_env, k))

    def _update_q_star(self):
        return self._q_star

    def _get_opt_actions(self):
        return np.argmax(self._q_star, axis=1) # the optimal action of each environment (num_env, )

    def step(self, a):
        assert len(a) == self.num_env
        assert np.isin(a, self.action_space).all()
        # update q values
        self._q_star = self._update_q_star()
        # get q values
        qs = self._q_star[np.arange(self.num_env), a] # (num_env, )
        # get rewards
        return np.random.normal(qs, 1, self.num_env) # (num_env, )

class KArmedNonstationaryEnv(KArmedEnv):
    r""" q_star with N(0, 0.01) random walk per step
    """
    def __init__(self, k=10, num_env=2000):
        super().__init__(k=k, num_env=num_env)
    
    def _init_q_star(self, num_env: int, k: int):
        return np.zeros((num_env, k), dtype=float)

    def _update_q_star(self):
        return self._q_star + np.random.normal(0, 0.01, self._q_star.shape)

class KArmedShiftedEnv(KArmedEnv):
    r""" q_star has mean +4 instead of 0
    """
    def __init__(self, k=10, num_env=2000):
        super().__init__(k=k, num_env=num_env)
    
    def _init_q_star(self, num_env: int, k: int):
        return np.random.normal(4, 1, (num_env, k))
