from RL_Optunity_CartPole.run_CartPole import run_cartpole  as optunity
import RL_Optunity_CartPole.RL_brain

from RL_Bayesian_CartPole.run_CartPole import run_cartpole as bayesian

def optimize(optimizer, **kwargs):
	if optimizer == "Optunity":
		optimize_optunity(**kwargs)
	elif optimizer == "Bayesian":
		optimize_bayesian(**kwargs)

def optimize_bayesian(meta_trials, optimisation_range,\
		reinforcement_learning_range,lr, rd):
	bayesian(meta_trials, optimisation_range, reinforcement_learning_range, lr, rd)

def optimize_optunity(n_evals, solver, lr, rd):
    solution, details, suggestion = \
    optunity.maximize(optunity, num_evals=n_evals, solver_name=solver,\
    	learning_rate=lr, reward_decay=rd)
    print("Solution:", solution)
    print("Details:", details)

#optimize("Optunity", n_evals=20, solver="cma-es", lr=[0.001, 0.03], ld=[0.01, 0.99])
optimize("Bayesian", meta_trials=10, optimisation_range=10,\
	reinforcement_learning_range=100, lr=[0.001, 0.1], rd=[0.01, 1])