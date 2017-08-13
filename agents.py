import itertools
from collections import defaultdict

import numpy as np
from scipy.stats import entropy

class BaseAgent:
  def __init__(self, env, num_episodes, *args):
    self.env = env
    self.num_episodes = num_episodes

  def learn(self):
    raise NotImplementedError(
      "Classes extending BaseAgent must implement learn")

class DefaultAgent(BaseAgent):
  def __init__(self, env, num_episodes):
    super().__init__(env, num_episodes)

    self.Ns = self.env.observation_space.n
    self.Na = self.env.action_space.n

    self.reset()

  def reset(self):
    Ns, Na = self.Ns, self.Na
    self.pi = np.ones((Ns, Na)) / Na
    self.Q = np.random.rand(Ns, Na)

    self.config = {
      "alpha1": 1e-2, # reward weight, trade-off between reward and exploration
      "alpha2": 1e-2, # learning rate for Q
      "lmbd": 1e3, # policy softening, trade-off between policy complexity and reward
      "epsilon": 1e-3 # threshold for breaking the blahut-arimoto style loop
    }

  def learn(self):
    env = self.env

    alpha1 = self.config["alpha1"]
    alpha2 = self.config["alpha2"]
    lmbd = self.config["lmbd"]
    epsilon = self.config["epsilon"]

    Ns, Na = self.Ns, self.Na

    Nxax = np.zeros((Ns, Na, Ns))
    pxax = np.ones((Ns, Na, Ns)) / Ns
    N = np.zeros((Ns, 1))

    paths = defaultdict(list)
    for episode in range(1, self.num_episodes+1):
      # (1) Initialization, get initial observation, etc.
      observation, done = env.reset(), False

      while not done:
        # (a) Update p_{t}(x) for all x in X
        # (the current estimate of the state visitation distribution)
        N[observation] += 1
        ptx = N / np.sum(N)

        # (b) Initialize q^{(0)}(a|x) for all a in A for all x in X
        qj0 = self.pi.copy()

        # (c) repeat until q^{j} converges
        qj1 = np.zeros_like(qj0)
        for j in itertools.count():
          pja = np.dot(qj0.T, ptx)
          pjx_ = np.tensordot(pxax, (qj0 * ptx)[:,:,np.newaxis], ((0,1), (0,1)))

          # TODO: remove
          # uncomment to verify probs and vectorized implementations
          # from pdb import set_trace; set_trace()
          # assert(np.allclose(np.sum(ptx), 1))
          # assert(np.allclose(np.sum(qj0), Ns))
          # assert(np.allclose(np.sum(pxax), Ns * Na))
          # assert(np.allclose(np.sum(pjx_), 1))
          # assert(np.allclose(np.sum(pja), 1))
          # assert(np.allclose(pjx_, loopy_pjx(pxax, qj0, ptx, Ns, Na)))

          qj1_unnorm = np.exp(
            entropy(pxax, pjx_[:,:,np.newaxis]).T / lmbd
            + alpha1 * self.Q
          ) * pja.T
          qj1 = qj1_unnorm / np.sum(qj1_unnorm, axis=1, keepdims=True)

          diff = np.sum(np.abs(qj1 - qj0))
          # if j % 1000 == 0: print(diff)
          if diff < epsilon: break

          qj0 = qj1

        self.pi = qj1.copy()

        # (d) Choose action, and obtain reward and next state
        action = np.argmax(self.pi[observation, :])
        new_observation, reward, done, _ = env.step(action)
        Nxax[observation, action, new_observation] += 1
        pxax[observation, action, :] = (Nxax[observation, action, :]
                                        / np.sum(Nxax[observation, action, :]))

        # TODO: check this
        # (e) Update the action-value function estimates Q.
        self.Q[observation, action] += alpha2 * (
          reward - self.Q[observation, action])

        paths[episode].append({
          "observation": observation,
          "action": action,
          "reward": reward
        })

        observation = new_observation

    return paths
