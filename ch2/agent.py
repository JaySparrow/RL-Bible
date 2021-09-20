import numpy as np

class EpsGreedyAgent:
    r""" use sample-average method for updating q value estimates
        k: number of actions
    """
    def __init__(self, eps: float=0, k: int=10, num_agt: int=2000, q_init: float=0):
        self.k = k
        self.num_agt = num_agt
        # action space
        self.action_space = np.arange(0, k)
        # degree of greediness & probabilities
        self.eps = np.clip(eps, 0, 1)
        self._opt_prob = 1 - self.eps + self.eps * 1/k
        self._rand_prob = self.eps * 1/k
        # estimates of q values
        self.Q = np.ones((num_agt, k), dtype=float) * q_init # (num_agt, k)
        self.N = np.zeros((num_agt, k), dtype=int) # (num_agt, k)
    
    def act(self):
        a = np.zeros(self.num_agt, dtype=int) 
        # probabilities
        prob = np.tile(self._rand_prob, (self.num_agt, self.k)) # (num_agt, k)
        prob[np.arange(self.num_agt), np.argmax(self.Q, axis=1)] = self._opt_prob
        # sample actions
        for i in range(self.num_agt):
            a[i] = np.random.choice(self.action_space, size=1, p=prob[i, :])
        return a # (num_agt, )

    def _step_size(self, idx: tuple):
        return 1 / self.N[idx]

    def learn(self, a, r):
        assert len(a) == len(r) == self.num_agt
        assert np.isin(a, self.action_space).all()
        # indeces
        idx = (np.arange(self.num_agt), a)
        # update counts for actions taken
        self.N[idx] += 1
        # update q value estimates
        self.Q[idx] = self.Q[idx] + self._step_size(idx) * (r - self.Q[idx])

class EpsGreedyConstStepAgent(EpsGreedyAgent):
    r""" use weighted-average method for updating q value estimates
    """
    def __init__(self, eps: float = 0, k: int = 10, num_agt: int = 2000, q_init: float = 0):
        super().__init__(eps=eps, k=k, num_agt=num_agt, q_init=q_init)
    
    def _step_size(self, idx: tuple):
        return 0.1

class UCBAgent:
    r""" use sample-average method for updating q value estimates
        k: number of actions
    """
    def __init__(self, c: float=0, k: int=10, num_agt: int=2000, q_init: float=0): # c=0 greedy agent
        self.k = k
        self.num_agt = num_agt
        # time step
        self.t = 0
        # action space
        self.action_space = np.arange(0, k)
        # degree of exploration
        self.c = c
        # estimates of q values
        self.Q = np.ones((num_agt, k), dtype=float) * q_init # (num_agt, k)
        self.N = np.zeros((num_agt, k), dtype=int) + 1e-6 # (num_agt, k)
    
    def act(self):
        self.t += 1
        criterion = self.Q + self.c * np.sqrt(np.log(self.t)/self.N) # (num_agt, k)
        return np.argmax(criterion, axis=1) # (num_agt, )

    def _step_size(self, idx: tuple):
        return 1 / self.N[idx]

    def learn(self, a, r):
        assert len(a) == len(r) == self.num_agt
        assert np.isin(a, self.action_space).all()
        # indeces
        idx = (np.arange(self.num_agt), a)
        # update counts for actions taken
        self.N[idx] += 1
        # update q value estimates
        self.Q[idx] = self.Q[idx] + self._step_size(idx) * (r - self.Q[idx])

class GradientAgent:
    def __init__(self, k: int=10, num_agt: int=2000, step_size: float=0.1, use_baseline: bool=True):
        self.k = k
        self.num_agt = num_agt
        self.step_size = step_size
        # action space
        self.action_space = np.arange(0, k)
        # action preference
        self.H = np.zeros((num_agt, k), dtype=float) # (num_agt, k)
        # baseline
        self.R = np.zeros(num_agt, dtype=float) # (num_agt, )
        self.N = 0
        if not use_baseline:
            self._baseline = lambda: np.zeros(num_agt, dtype=float) # (num_agt, )
        else:
            self._baseline = lambda: self.R / self.N # (num_agt, )

    def _get_policy(self):
        return np.exp(self.H) / np.sum(np.exp(self.H), axis=1, keepdims=True) # (num_agt, k)

    def act(self):
        # calc policy, i.e. action probability distributions
        policy = self._get_policy() # (num_agt, k)
        # sample actions
        a = np.zeros(self.num_agt, dtype=int) 
        for i in range(self.num_agt):
            a[i] = np.random.choice(self.action_space, size=1, p=policy[i, :])
        return a # (num_agt, )

    def learn(self, a, r):
        assert len(a) == len(r) == self.num_agt
        assert np.isin(a, self.action_space).all()
        # update baseline
        self.R += r
        self.N += 1
        # indeces
        idx = (np.arange(self.num_agt), a)
        # update preferences
        H      = self.H      - self.step_size * (r - self._baseline()).reshape((-1, 1)) * self._get_policy() # (num_agt, k)
        H[idx] = self.H[idx] + self.step_size * (r - self._baseline()) * (1 - self._get_policy()[idx])
        self.H = H



        