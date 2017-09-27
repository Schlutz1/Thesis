from RL_Optunity_CartPole.run_CartPole import run_cartpole as opt_cartpole
import RL_Optunity_CartPole.RL_brain
from RL_Bayesian_CartPole.run_CartPole import run_cartpole as bayesian_cartpole
from RL_Bayesian_CartPole.run_MountainCar import run_mountaincar as bayesian_mountaincar
from RL_Bayesian_Bandit.Bandit import run_bandit as bayesian_bandit

import optunity

def optimize(optimizer, challenge, **kwargs):
    if optimizer == "Optunity":
        optimize_optunity(**kwargs)
    elif optimizer == "Bayesian":
        optimize_bayesian(challenge, **kwargs)

def optimize_bayesian(problem, **kwargs):
    if problem == "Bandit":
        bayesian_bandit(**kwargs)
    elif problem == "Mountain Car":
        bayesian_mountaincar(**kwargs)
    elif problem == "Cart Pole":
        bayesian_cartpole(**kwargs)

def optimize_optunity(**kwargs):
    hps,_,_ = optunity.maximize(**kwargs)
    print(hps)


#optimize("Optunity", None, f=opt_cartpole, num_evals=200, learn_rate=[0.001, 0.1], rew_decay=[0.01, 1])
#optimize("Bayesian", "Bandit", meta_trials=100, n_iter=200, lr=[0.001, 0.1], n_episodes=20, bandits=[0.5, 0, -0.3, -0.6], e=0.2)
optimize("Bayesian", "Cart Pole", meta_trials=100, optimisation_range=50, reinforcement_learning_range=50,\
    learning_rate_range=[0.001, 0.1], reward_decay_range=[0.01, 1])
#optimize("Bandit", n_iter=10, lr=[0.0001, 1], n_episodes=1000,\
    #bandits=[0.5, 0, -0.3, -0.6], e=0.2)
