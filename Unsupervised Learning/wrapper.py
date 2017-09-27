from RL_Optunity_CartPole.run_CartPole import run_cartpole  as optunity
import RL_Optunity_CartPole.RL_brain

def optimize_optunity(n_evals, solver, lr, rd):
    solution, details, suggestion = \
    optunity.maximize(optunity, num_evals=n_evals, solver_name=solver, learning_rate=lr, reward_decay=rd)
    print("Solution:", solution)
    print("Details:", details)

optimize(20, "cma-es", [0.001, 0.03], [0.01, 0.99])
