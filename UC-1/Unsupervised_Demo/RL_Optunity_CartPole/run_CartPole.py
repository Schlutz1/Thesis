import gym
from .RL_brain import PolicyGradient
from writeFunction import write_function
import optunity

def run_cartpole(learn_rate, rew_decay):
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
        learning_rate=learn_rate,
        reward_decay=rew_decay,
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
                break
            observation = observation_
    write_function("cartpole_meta_analysis.csv", "Reinforcement Learning", "algo_type", "_", "2", "Cart Pole",\
        "iteration", learn_rate, rew_decay, Total_Rewards)
    return Total_Rewards
