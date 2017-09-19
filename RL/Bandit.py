# Cartpole Implementation

from neupy import algorithms, layers
import tensorflow as tf
import numpy as np
import random
import math
import gym
import sys

from gaussianProcess import gaussian_process, vector_2d
from surrogateFunction import next_parameter_by_ei


def optimise_learning_rate():
    return "fuck"


def pullBandit(bandit):
    # Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        # return a positive reward.
        return 1
    else:
        # return a negative reward.
        return -1


def bandit_function(total_episodes, learning_rate, e, bandits):

    num_bandits = len(bandits)
    total_reward = np.zeros(num_bandits)

    tf.reset_default_graph()

    # These two lines established the feed-forward part of the network. This
    # does the actual choosing.
    weights = tf.Variable(tf.ones([num_bandits]))
    chosen_action = tf.argmax(weights, 0)

    # tf instantiation
    reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
    action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
    responsible_weight = tf.slice(weights, action_holder, [1])
    loss = -(tf.log(responsible_weight)*reward_holder)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        while i < total_episodes:

            # Choose either a random action or one from our network.
            if np.random.rand(1) < e:
                action = np.random.randint(num_bandits)
            else:
                action = sess.run(chosen_action)

            # Get our reward from picking one of the bandits.
            reward = pullBandit(bandits[action])

            # Update the network.
            _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={
                reward_holder: [reward], action_holder: [action]})

            # Update our running tally of scores.
            total_reward[action] += reward
            if i % 50 == 0:
                print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)
            i += 1
    print "The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising...."
    if np.argmax(ww) == np.argmax(-np.array(bandits)):
        print "...and it was right!"
    else:
        print "...and it was wrong!"


# intialisation
if __name__ == '__main__':

        # problem space definition
    bandits = [0.5, 0, -0.3, -0.5]

    # hyperparameter defintions
    total_episodes = 1000  # Set total number of episodes to train agent on.
    learning_rate = 0.001  # Learning rate of model
    e = 0.2  # Set the chance of taking a random action.

    bandit_function(total_episodes, learning_rate, e, bandits)
