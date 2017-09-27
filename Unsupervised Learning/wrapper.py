from RL_Optunity_CartPole.run_CartPole import run_cartpole as opt
import RL_Optunity_CartPole.RL_brain

from RL_Bayesian_CartPole.run_CartPole import run_cartpole as bayesian_cp
from RL_Bayesian_CartPole.run_MountainCar import run_mountaincar as bayesian_mc

from RL_Bayesian_Bandit.Bandit import run_bandit as bayesian_bandit

import optunity

def preprocess():
    return 0

def optimize(optimizer, **kwargs):
    if optimizer == "Optunity":
        optimize_optunity(**kwargs)
    elif optimizer == "Bayesian":
        optimize_bayesian(**kwargs)

def optimize_bayesian(problem, **kwargs):
    if problem == "Bandit":
        bayesian_bandit(**kwargs)
    elif problem == "Mountain Car":
        bayesian_mc(**kwargs)
    elif problem == "Cart Pole":
        bayesian_cp(**kwargs)

def optimize_optunity(n_evals, solver, **kwargs):
    solution, details, suggestion = \
    optunity.maximize(opt, num_evals=n_evals, solver_name=solver,\
        **kwargs)
    print(solution)


optimize_optunity(200, "cma-es", lr=[0.001, 0.03], rd=[0.01, 0.99])
#optimize("Bayesian", meta_trials=10, optimisation_range=10,\
#   reinforcement_learning_range=100, lr=[0.001, 0.1], rd=[0.01, 1])
#optimize("Bayesian", lr=0.02, reward_decay=0.995)
#optimize("Bandit", n_iter=10, lr=[0.0001, 1], n_episodes=1000,\
    #bandits=[0.5, 0, -0.3, -0.6], e=0.2)
