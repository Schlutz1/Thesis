"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from .rlBrain import policy_gradient

# lr: learning rate
# rd: reward decay
def run_mountaincar(lr, rd, n_episodes):
    DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
    # episode: 154   reward: -10667
    # episode: 387   reward: -2009
    # episode: 489   reward: -1006
    # episode: 628   reward: -502

    RENDER = True  # rendering wastes time

    env = gym.make('CartPole-v0')
    env.seed(1)     # reproducible, general Policy gradient has high variance
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=lr,
        reward_decay=rd,
        # output_graph=True,
    )

    for i_episode in range(n_episodes):

        observation = env.reset()

        while True:
            if RENDER: env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)     # reward = -1 in all cases

            RL.store_transition(observation, action, reward)

            if done:
                # calculate running reward
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering

                print("episode:", i_episode, "  reward:", int(running_reward))

                vt = RL.learn()  # train

                break

            observation = observation_

    write_function(filename, "Reinforcement Learning", "Gaussian", "RLBrain", "2", "CartPole",
        opt_iteration,learning_rate , 0.99,bayesian_cost_value)
