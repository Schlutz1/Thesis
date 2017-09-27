# Cartpole Implementation

from neupy import algorithms, layers
import tensorflow as tf
import numpy as np
import itertools
import random
import math
import gym
import sys
import csv

# Note, you may treat these as black box functions as they are entirely
# statistical methods
from .gaussianProcess import gaussian_process, vector_2d
from .gaussianProcess import next_parameter_by_ei


def writeFunction(trial_number, learning_rate, final_score) :
	outputString = "bandit_meta_analysis.csv"
	with open(outputString,'wb') as fin: 
			testArray = (trial_number, learning_rate, final_score)
			fieldNames = ['trial_number', 'learning_rate', 'final_score']
			writer = csv.writer(fin)
			testArray = zip(*testArray)
			writer.writerow(fieldNames)
			for i in range(len(trial_number)):
				writer.writerow(testArray[i])

def pullBandit(bandit):
	# Get a random number.
	result = np.random.randn(1)
	if result > bandit:
		# return a positive reward.
		return 1
	else:
		# return a negative reward.
		return -1


def bandit_function(total_episodes, learning_rate, e, bandits, noramlise_reward):

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
				print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
			i += 1
	print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
	
	reward_value = [a*b for a,b in zip(total_reward, noramlise_reward)]
	return (sum(reward_value))

# intialisation
def run_bandit(n_iter, lr, n_episodes, bandits, e):
	# problem space
	noramlise_reward = np.absolute(bandits)
	

	min_learning_rate, max_learning_rate = lr
	learning_rate_choices = np.arange(min_learning_rate, max_learning_rate + 1)

	normalisation_factor = min_learning_rate
	min_learning_rate, max_learning_rate = min_learning_rate / \
		normalisation_factor, max_learning_rate / normalisation_factor

	meta_trials = 100

	trial_number = []
	learning_rate_array = []
	final_score = []

	for i in range(meta_trials) :
		scores = []
		parameters = []

		trial_number.append(i)

		for iteration in range(1, n_iter + 1):  # gaussian iteration process starts here
			print("this is iteration number: " + str(i) + "_" + str(iteration))
			if iteration in (1, 2):
				learning_rate = random.randint(
					min_learning_rate, max_learning_rate)
				learning_rate = learning_rate * normalisation_factor
			print(learning_rate)

			res = bandit_function(n_episodes, learning_rate, e, bandits, noramlise_reward)

			print("Resultant score: " + str(res))

			scores.append(res)
			parameters.append(learning_rate)

			print(scores)
			print(parameters)

			if iteration < 2:
				continue  # need two samples to actually minimise

			
			y_mean, y_std = gaussian_process(parameters, scores, learning_rate_choices)
			 
			y_min = min(scores)

			learning_rate = next_parameter_by_ei(y_min, y_mean, y_std, learning_rate_choices)

			if y_min == 0 or learning_rate in parameters :
				break #lowest expectation achieved
			
		
		max_score_index = np.argmax(scores)
		final_score.append(scores[max_score_index])
		learning_rate_array.append(parameters[max_score_index])
		
		print(parameters[max_score_index]) # final learning rate

	optimal_learning_rate = max(set(parameters), key=parameters.count)
	print("Optimal learning rate : " + str(optimal_learning_rate))

	writeFunction(trial_number, learning_rate_array, final_score)



