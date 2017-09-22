import gym
from RL_brain import PolicyGradient
from writeFunction import write_function
import matplotlib.pyplot as plt
import optunity

def run_cartpole(learning_rate, reward_decay):
    print("LR:", learning_rate," RD:",reward_decay)
    Total_Rewards = 0 # Total Rewards
    DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
    RENDER = False  # rendering wastes time
    env = gym.make('CartPole-v0')
    env.seed(1)     # reproducible, general Policy gradient has high variance
    env = env.unwrapped
    # print(env.action_space)
    # print(env.observation_space)
    # print(env.observation_space.high)
    # print(env.observation_space.low)
    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.99,
        # output_graph=True,
    )
    for i_episode in range(30):
        observation = env.reset()
        while True:
            #env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_transition(observation, action, reward)
            if done:
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
                # print("episode:", i_episode, "  reward:", int(running_reward))
                Total_Rewards += running_reward
                vt = RL.learn()
                # if i_episode == 0:
                #     plt.plot(vt)    # plot the episode vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break
            observation = observation_
    print(Total_Rewards,"\n")
    return Total_Rewards

# By default, num_particles=10, num_generations=5
solution, details, suggestion = optunity.maximize(run_cartpole, num_evals=20, solver_name="particle swarm", learning_rate=[0.001, 0.03], reward_decay=[0.01, 0.99])
print("Solution:", solution)
print("Details:", details)
# print("Optimum:", details[0])
# print("Iterations and Time:", details[1])
# print("Call_Log:", details[2])
# print(len(details))
# # print(details[""])
print("Suggestion:", suggestion)
#write_function(trial_number, learning_rate, reward_decay, final_score)
