import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import sys
import json
import argparse
from sklearn.feature_extraction.text import HashingVectorizer

# Edited 20/04 by Group B
# Changes:  1. streamline computational graph construction
#           2. feed_dicts for training
#           3. Competitive performance on benchmark datasets

class ml_metadata:

    def __init__(self, features=None, features_numeral=None, classifier=None, options=None):
        self.features = features  # feature list
        self.features_numeral = features_numeral
        self.classifier = classifier  # class
        self.pipeFilename = os.path.basename(__file__).strip(".pyc")
        self.name = None
        options_default = {"r_regexp": "",
                           "alpha": 0.001,
                           "epochs": 5000,
                           "name": "default_model_name",
                           "n_features": 256,
                           "arch": [128, 128, 64],
                           "testtrainratio": 0.3
                           }
        if options != None:
            self.setOptionsJSON(options)
            for opt in self.options.keys():
                if opt is 'classifier_func':
                    for sub_opt in opt:
                        options_default[sub_opt] = self.options[opt][sub_opt]
                else:
                    options_default[opt] = self.options[opt]

        # SET THESE DYNAMICALLY THROUGH SPLUNK!
        self.r_regexp = str(options_default["r_regexp"])
        self.alpha = float(options_default["alpha"])
        self.steps = int(options_default["epochs"])
        self.model_name = str(options_default["name"])
        self.n_features = int(options_default["n_features"])
        self.arch = options_default['arch']
        self.testtrainratio = options_default["testtrainratio"]

    def setOptionsJSON(self, options):
        self.options = json.loads(options)
        print self.options
        return unicode(self.options)


class ml_pipe:

    def __init__(self, data, model_name, architecture, features_numeral, features, classifier, load=False, options='{"isRegression":false}'):
        # Alpha -> Beta for AdamOptimizer
        self.model = model_name
        self.tensor_df = pd.DataFrame()
        self.data = data
        self.features_numeral = features_numeral

        self.features = features
        self.classifier = classifier

        # SET THESE DYNAMICALLY THROUGH SPLUNK!
        # Define network architecture - make this user input!
        #ideal arch self.architecture = [128, 128, 64]
        self.architecture = architecture
        self.batch_size = 128  # Training batch size
        self.val_batch_size = 32  # Validation batch size
        # self.beta = 1e-4
        self.metadata = ml_metadata(features, classifier, options)
        # Hashing for bag of words
        self.hv = HashingVectorizer(
            decode_error='ignore', n_features=self.metadata.n_features)
        self.modelbasedir = "./models/"
        self.df_X, self.df_y = self.curate_data(self.data)

        # Get number of features, classes
        self.n_features = self.df_X.values.shape[1]
        self.n_classes = self.df_y.values.shape[1]

        # Build the computational graph or make predictions
        if load == False:
            # Split data into training, test, validation sets at 0.8:0.1:0.1
            # ratio - make this user input!
            self.train_test_split(self.df_X, self.df_y, holdout_size=0.1)
            self.graph = tf.Graph()
            self.create_graph()

            # Execute the training graph 
            self.final_score = self.execute_training_graph()
        else:
            self.graph = tf.Graph()
            self.create_graph()

            # Execute the prediction graph
            self.execute_prediction_graph()

    def get_final_score(self):
        return self.final_score

    def curate_data(self, data):
        #df_Xpf, df_y = data[data.columns[:-1]], pd.get_dummies(data[data.columns[-1]])

        df_X = pd.DataFrame()
        df_y = pd.DataFrame()

        self.tcount = 0
        for col in data.columns:
            if col in self.features:
                self.tcount = self.tcount + 1
                labels = [col + "_" + str(x)
                          for x in range(self.metadata.n_features)]
                df_X = pd.concat([df_X, pd.DataFrame(data=self.hv.transform(data[col]).toarray(), columns=labels)],
                                 axis=1)
            elif col in self.features_numeral:
                df_X = pd.concat(
                    [df_X, pd.DataFrame({col: data[col]})], axis=1)
            elif col in self.classifier:
                df_y = pd.get_dummies(data[col])

        df_X = df_X.sort_index(axis=1)
        return df_X, df_y

    def train_test_split(self, features, labels, holdout_size=0.2, val_size=0.5):
        from sklearn.model_selection import train_test_split
        # Split data into training and test sets
        X_all = features.values
        y_all = labels.values
        self.X_train, self.X_holdout, self.y_train, self.y_holdout = train_test_split(
            X_all, y_all, test_size=holdout_size, random_state=24601)
        self.X_test, self.X_val, self.y_test, self.y_val = train_test_split(
            self.X_holdout, self.y_holdout, test_size=val_size, random_state=24601)

    ############################################
    #  Helper functions for graph construction
    ############################################

    def layer_weights(self, shape):
        # Return weight tensor of given shape using Xavier initialization
        W = tf.get_variable("weights", shape=shape,
                            initializer=tf.contrib.layers.xavier_initializer())
        return W

    def layer_biases(self, shape):
        # Return bias tensor of given shape with small initialized constant
        # value
        b = tf.get_variable("biases", shape=shape,
                            initializer=tf.constant_initializer(0.01))
        return b

    def dropout_op(self, layer, keep_prob):
        # Neurons in given layer have retention probability keep_prob to combat
        # overfitting
        fc_drop = tf.nn.dropout(layer, keep_prob)
        return fc_drop

    def hidden_layer_ops(self, x, shape, name, keep_prob, activation=tf.nn.relu):
        # Add operations to graph to construct hidden layers
        with tf.variable_scope(name) as scope:
            # scope.reuse_variables() # otherwise tf.get_variable() checks that
            # already existing vars are not shared by accident
            weights = self.layer_weights(shape=shape)
            biases = self.layer_biases(shape=[shape[1]])

            # Apply non-linearity. Default is ReLU
            actv = activation(tf.matmul(x, weights) + biases)
            layer_output = self.dropout_op(actv, keep_prob)

        return layer_output

    def readout_ops(self, x, shape, name, keep_prob, activation=tf.nn.relu):
        # Don't apply non-linearity, dropout on output layer
        with tf.variable_scope(name) as scope:
            weights = self.layer_weights(shape=shape)
            biases = self.layer_biases(shape=[shape[1]])
            layer_output = tf.matmul(x, weights) + biases

            # TensorBoard monitoring
            tf.summary.histogram('readout', layer_output)

        return layer_output

    def prediction_operation(self, y_true, y_pred):
        # Evaluate accuracy of predicted data
        correct_prediction = tf.equal(
            tf.argmax(y_true, 1), tf.argmax(y_pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def data_feed_iterator(self, features, labels, batch_size):
        # Takes raw data as numpy arrays and feeds into computation graph
        # Input data has shape [batch_size, self.n_features]
        while True:
            # Shuffle labels and features: shuf_features = features[idxs],
            # shuf_labels = labels[idxs]
            idxs = np.arange(0, features.shape[0])
            np.random.shuffle(idxs)
            if batch_size > len(idxs):
                batch_size = len(idxs)
            random_batch_idx = np.random.choice(
                idxs, size=batch_size, replace=False)

            x_batch = features[random_batch_idx]  # .astype('float32')
            labels_batch = labels[random_batch_idx]
            #onehot_labels_batch = tf.one_hot(indices=tf.cast(labels_batch, tf.int32), depth=2)

            return [x_batch, labels_batch]

    def epoch_shuffle(self, features, labels):
        # Shuffles features, labels at end of epoch
        rand_idx = np.random.permutation(features.shape[0])
        features = features[rand_idx]
        labels = labels[rand_idx]

        return features, labels

    def create_inference_graph(self):
        # Builds the graph to run inference

        with self.graph.as_default():
            # A more modular way to build the graph
            # Accept variable batch size
            # change n_features to number of input features
            self.tf_X = tf.placeholder(
                tf.float32, shape=[None, self.n_features])
            self.tf_labels = tf.placeholder(
                tf.float32, shape=[None, self.n_classes])
            self.keep_prob = tf.placeholder(tf.float32)
            self.hidden_layers = [self.tf_X]

            # Use 3 hidden layers we can wrap this in a for loop for arbitrary # of layers
            # Use self.architecture as number of nodes in each layer, in
            # general more nodes -> better performance
            self.hidden_layers.append(self.hidden_layer_ops(self.hidden_layers[0], shape=[
                self.n_features, self.architecture[0]], name='hashlayer', keep_prob=self.keep_prob))
            for i in range(1, len(self.architecture)-1):
                hidden_n = self.hidden_layer_ops(self.hidden_layers[i], shape=[self.architecture[
                    i], self.architecture[i+1]], name='hidden' + str(i), keep_prob=self.keep_prob)
                self.hidden_layers.append(hidden_n)

        # Add readout layer - output layer shape is the number of classes we
        # predict
            self.readout = self.readout_ops(self.hidden_layers[-1], shape=[
                                            self.architecture[-1], self.n_classes], name='readout', keep_prob=1.0)

            self.prediction_op = tf.nn.softmax(self.readout)
            self.classification_op = tf.argmax(self.prediction_op, 1)

    def create_graph(self):
        # Adds necessary operations to computational graph to be evaluated
        # during training

        with self.graph.as_default():
            # A more modular way to build the graph
            # Accept variable batch size
            # change n_features to number of input features
            self.tf_X = tf.placeholder(
                tf.float32, shape=[None, self.n_features])
            self.tf_labels = tf.placeholder(
                tf.float32, shape=[None, self.n_classes])
            self.keep_prob = tf.placeholder(tf.float32)
            self.hidden_layers = [self.tf_X]

            # arb layer construction
            self.hidden_layers.append(self.hidden_layer_ops(self.hidden_layers[0], shape=[
                self.n_features, self.architecture[0]], name='hashlayer', keep_prob=self.keep_prob))
            for i in range(1, len(self.architecture)-1):
                hidden_n = self.hidden_layer_ops(self.hidden_layers[i], shape=[self.architecture[
                    i], self.architecture[i+1]], name='hidden' + str(i), keep_prob=self.keep_prob)
                self.hidden_layers.append(hidden_n)

        # Add readout layer - output layer shape is the number of classes we
        # predict
            self.readout = self.readout_ops(self.hidden_layers[-1], shape=[
                                            self.architecture[-1], self.n_classes], name='readout', keep_prob=1.0)

            # Use X-Entropy loss, add to TensorBoard for visualization
            self.cross_entropy_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.tf_labels, logits=self.readout))
            tf.summary.scalar('cross_entropy', self.cross_entropy_op)

            # Use AdamOptimizer as recommended by TensorFlow devs, experiment
            # with beta learning rate
            self.train_op = tf.train.AdamOptimizer(
                1e-4).minimize(self.cross_entropy_op, name='train_op')

        # Make predictions (probabilities) and classifications (categorical)
            self.prediction_op = tf.nn.softmax(self.readout)
            self.classification_op = tf.argmax(self.prediction_op, 1)

        # Get accuracies
            self.accuracy = self.prediction_operation(
                y_true=self.tf_labels, y_pred=self.readout)

            self.saver = tf.train.Saver()

    def execute_training_graph(self):
        # Could train using feed_dicts, but much more efficient to use
        # queues...
        self.steps = 1000
        self.epochs = 50
        self.position = 0
        self.dropout_prob = 0.85

        # Epochs better metric for network training time
        with tf.Session(graph=self.graph) as sess:
            v_acc_best = 0.0
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            print "Beginning model training:\n"
            for step in range(self.steps):
                # Feed data in using feed_dicts - simple but slow
                # offset = (step * self.batch_size) % (self.data_class -
                # self.batch_size) #GET DATA_CLASS LENGTH

                if ((self.tf_X.shape[0] - self.position) > self.batch_size):
                    self.batch_train = self.tf_X[
                        self.position:self.position+self.batch_size], self.tf_labels[self.position:self.position+self.batch_size]
                    self.position += self.batch_size
                else:
                    # Epoch completed, reshuffle training batches + labels
                    self.batch_train = [self.X_train[
                        self.position:], self.y_train[self.position:]]
                    self.position = 0
                    # SHUFFLE TENSORS HERE
                    self.X_train, self.y_train = self.epoch_shuffle(
                        self.X_train, self.y_train)
                    # print "End of epoch.\n"
                print self.batch_train[0].shape
                # Feed in training data and labels to graph
                self.feed_dict_train = {self.tf_X: self.batch_train[
                    0], self.tf_labels: self.batch_train[1], self.keep_prob: self.dropout_prob}

                # Inject data into tensors in computation graph
                self.t_step, self.t_error = sess.run(
                    [self.train_op, self.cross_entropy_op], feed_dict=self.feed_dict_train)

                # Periodically print training diagnostics every 100 steps
                if (step % 100 == 0):
                    improved_flag = ''
                    self.batch_test = self.data_feed_iterator(
                        self.X_test, self.y_test, batch_size=self.val_batch_size)
                    self.feed_dict_val = {self.tf_X: self.batch_test[
                        0], self.tf_labels: self.batch_test[1], self.keep_prob: 1.0}
                    t_acc = sess.run(
                        self.accuracy, feed_dict=self.feed_dict_train)
                    v_acc = sess.run(
                        self.accuracy, feed_dict=self.feed_dict_val)

                    if v_acc > v_acc_best:
                        v_acc_best = v_acc
                        improved_flag = '*'
                        if (float(step)/self.steps > 0.5):
                            save_path = self.saver.save(sess, os.path.join(
                                self.modelbasedir, self.model + '_tf_pipe_best.ckpt'))

                    delta_t = time.time() - start_time
                    print "Step: %d\t|\tTraining accuracy: %g\t|\tValidation Accuracy: %g (%.3f s) %s" % (step, t_acc, v_acc, delta_t, improved_flag)

            # Training complete, save model and report final accuracies

            save_path = self.saver.save(sess, os.path.join(
                self.modelbasedir, self.model + '_tf_pipe.ckpt'), global_step=step)
            print "Model saved at: %s" % (save_path)
            delta_t = time.time() - start_time
            final_train_accuracy = self.accuracy.eval(
                feed_dict={self.tf_X: self.X_train, self.tf_labels: self.y_train, self.keep_prob: 1.0})
            final_test_accuracy = self.accuracy.eval(
                feed_dict={self.tf_X: self.X_test, self.tf_labels: self.y_test, self.keep_prob: 1.0})

            print "Training complete! Time elapsed: %g s\n" % (delta_t)
            print "Steps: %d, epochs: %d\n" % (self.steps, self.steps*self.batch_size/(self.X_train.shape[0]))
            print "Train accuracy: %g\nValidation accuracy: %g\n" % (final_train_accuracy, final_test_accuracy)
            return final_test_accuracy

    def execute_prediction_graph(self):
        """
        Classification on a new instance is given by the softmax of the output of the final readout layer.
        """
        # Load saved graph metadata in current default graph, restore in
        # session
        checkpoint_files = tf.train.latest_checkpoint(self.modelbasedir)
        saver = tf.train.import_meta_graph(checkpoint_files + '.meta')
        graph = tf.get_default_graph()
        print self.df_X
        with tf.Session(graph=self.graph) as sess:
            start_time = time.time()

            # Restore the trained model
            saver.restore(sess, checkpoint_files)
            print "Model %s restored." % (checkpoint_files)

            feed_dict_predict = {
                self.tf_X: self.df_X.values, self.keep_prob: 1.0}
            self.NN_output, self.classifications = sess.run(
                [self.prediction_op, self.classification_op], feed_dict=feed_dict_predict)

        np.save(os.path.join(self.modelbasedir, self.model +
                             '_softmax_pred.npy'), self.NN_output)
        np.save(os.path.join(self.modelbasedir, self.model +
                             '_y_pred.npy'), self.classifications)

        delta_t = time.time() - start_time
        print "Inference complete. Duration: %g s. Results saved in: %s" % (delta_t, self.modelbasedir)

def client_model(architecture, filepath): #use if called by optimizing wrapper
#if __name__ == "__main__" : #use if standalone

    parser = argparse.ArgumentParser(
        description='Train/make predictions using a neural network model.')
    parser.add_argument('data_file', type=str,
                        help='Data to train/make predictions on in .csv format')
    parser.add_argument('-p', '--predict', dest='predict', required=False, action='store_true',
                        help='Flag to make predictions. Otherwise we train the network.')
    args = parser.parse_args()

    print "Loading %s" % (filepath)
    path2csv = os.path.join(
        '/Users/ms_acn/Documents/Thesis/UC-3/lookups', filepath)
    features = pd.read_csv(path2csv).fillna('')
    model_name = os.path.splitext(args.data_file)[0]

    if args.predict:
        print "Inference from previously trained network."
        ml = ml_pipe(data=features, model_name=model_name, architecture=architecture, features_numeral='', features=[
                'problem_abstract', 'asset_id'], classifier='Category', load=True, options='{"isRegression":false}')
    else:
        print "Training neural network on dataset"
        ml = ml_pipe(data=features, model_name=model_name, architecture=architecture, features_numeral='', features=[
                'problem_abstract', 'asset_id'], classifier='Category', load=False, options='{"isRegression":false}')

        print "Classifier trained, model saved. Bye!\n"

    return ml.get_final_score()
