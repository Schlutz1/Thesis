from RL_Optunity_CartPole.run_CartPole import run_cartpole  as optunity
import RL_Optunity_CartPole.RL_brain

from RL_Bayesian_CartPole.run_CartPole import run_cartpole as bayesian

def optimize(optimizer, **kwargs):
	if optimizer == "Optunity":
		optimize_optunity(**kwargs)
	elif optimizer == "Bayesian":
		optimize_bayesian(10, 10, 100, [0.001, 0.1], [0.01, 1])

def optimize_bayesian(meta_trials, optimisation_range,\
		reinforcement_learning_range,lr, rd):
	bayesian()

def optimize_optunity(n_evals, solver, lr, rd):
    solution, details, suggestion = \
    optunity.maximize(optunity, num_evals=n_evals, solver_name=solver,\
    	learning_rate=lr, reward_decay=rd)
    print("Solution:", solution)
    print("Details:", details)

optimize("Optunity", n_evals=20, solver="cma-es", lr=[0.001, 0.03], ld=[0.01, 0.99])