# Jeremy Pretty
# CSC 580 Crit 5 Option 1
import numpy as np
import matplotlib.pyplot as plt
import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(456)

# Load Tox21 dataset
_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

# Generate TensorFlow graph using a random forest classifier
sklearn_model = RandomForestClassifier(class_weight="balanced", n_estimators=50)
print("About to fit model on train set.")
sklearn_model.fit(train_X, train_y)
train_y_pred = sklearn_model.predict(train_X)
valid_y_pred = sklearn_model.predict(valid_X)
test_y_pred = sklearn_model.predict(test_X)

weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
print("Weighted train Classification Accuracy: %f" % weighted_score)
weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Weighted valid Classification Accuracy: %f" % weighted_score)
weighted_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
print("Weighted test Classification Accuracy: %f" % weighted_score)


def eval_tox21_hyperparams(n_hidden_values=[50], n_layers_values=[1], learning_rate_values=[0.001],
                           dropout_prob_values=[0.5], n_epochs_values=[45], batch_size_values=[100],
                           weight_positives_values=[True], num_repeats=3):
    """
    Evaluate different combinations of hyperparameters for Tox21 model.

    Args:
        n_hidden_values (list): List of values for the number of hidden units.
        n_layers_values (list): List of values for the number of layers.
        learning_rate_values (list): List of values for the learning rate.
        dropout_prob_values (list): List of values for the dropout probability.
        n_epochs_values (list): List of values for the number of epochs.
        batch_size_values (list): List of values for the batch size.
        weight_positives_values (list): List of values for the weight_positives parameter.
        num_repeats (int): Number of times to repeat evaluation for each hyperparameter combination.

    Returns:
        None
    """
    best_accuracy = 0.0
    best_hyperparameters = None

    for n_hidden in n_hidden_values:
        for n_layers in n_layers_values:
            for learning_rate in learning_rate_values:
                for dropout_prob in dropout_prob_values:
                    for n_epochs in n_epochs_values:
                        for batch_size in batch_size_values:
                            for weight_positives in weight_positives_values:
                                accuracy_sum = 0.0

                                for _ in range(num_repeats):
                                    accuracy = evaluate_model(n_hidden, n_layers, learning_rate, dropout_prob,
                                                              n_epochs, batch_size, weight_positives)
                                    accuracy_sum += accuracy

                                average_accuracy = accuracy_sum / num_repeats

                                if average_accuracy > best_accuracy:
                                    best_accuracy = average_accuracy
                                    best_hyperparameters = (n_hidden, n_layers, learning_rate, dropout_prob,
                                                            n_epochs, batch_size, weight_positives)

    print("Best hyperparameters:")
    print("n_hidden = %d" % best_hyperparameters[0])
    print("n_layers = %d" % best_hyperparameters[1])
    print("learning_rate = %f" % best_hyperparameters[2])
    print("dropout_prob = %f" % best_hyperparameters[3])
    print("n_epochs = %d" % best_hyperparameters[4])
    print("batch_size = %d" % best_hyperparameters[5])
    print("weight_positives = %s" % str(best_hyperparameters[6]))
    print("Best validation accuracy: %f" % best_accuracy)


def evaluate_model(n_hidden, n_layers, learning_rate, dropout_prob, n_epochs, batch_size, weight_positives):
    """
    Evaluate the Tox21 model with given hyperparameters.

    Args:
        n_hidden (int): Number of hidden units.
        n_layers (int): Number of layers.
        learning_rate (float): Learning rate.
        dropout_prob (float): Dropout probability.
        n_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        weight_positives (bool): Whether to weight positive examples.

    Returns:
        float: Weighted classification accuracy on the validation set.
    """
    d = 1024
    graph = tf.Graph()

    with graph.as_default():
        # Load Tox21 dataset
        _, (train, valid, test), _ = dc.molnet.load_tox21()
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w

        # Remove extra tasks
        train_y = train_y[:, 0]
        valid_y = valid_y[:, 0]
        test_y = test_y[:, 0]
        train_w = train_w[:, 0]
        valid_w = valid_w[:, 0]
        test_w = test_w[:, 0]

        with tf.name_scope("placeholders"):
            x = tf.placeholder(tf.float32, (None, d))
            y = tf.placeholder(tf.float32, (None,))
            w = tf.placeholder(tf.float32, (None,))
            keep_prob = tf.placeholder(tf.float32)

        for layer in range(n_layers):
            with tf.name_scope("layer-%d" % layer):
                W = tf.Variable(tf.random_normal((d, n_hidden)))
                b = tf.Variable(tf.random_normal((n_hidden,)))
                x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
                x_hidden = tf.nn.dropout(x_hidden, keep_prob)

        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal((n_hidden, 1)))
            b = tf.Variable(tf.random_normal((1,)))
            y_logit = tf.matmul(x_hidden, W) + b
            y_one_prob = tf.sigmoid(y_logit)
            y_pred = tf.round(y_one_prob)

        with tf.name_scope("loss"):
            y_expand = tf.expand_dims(y, 1)
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)

            if weight_positives:
                w_expand = tf.expand_dims(w, 1)
                entropy = w_expand * entropy

            l = tf.reduce_sum(entropy)

        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)

        with tf.name_scope("summaries"):
           tf.summary.scalar("loss", l)
        merged = tf.summary.merge_all()

        hyperparam_str = "d-%d-hidden-%d-lr-%f-n_epochs-%d-batch_size-%d-weight_pos-%s" % (
            d, n_hidden, learning_rate, n_epochs, batch_size, str(weight_positives))

        train_writer = tf.summary.FileWriter('/tmp/fcnet-func-' + hyperparam_str, tf.get_default_graph())

        N = train_X.shape[0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            step = 0

            for epoch in range(n_epochs):
                pos = 0

                while pos < N:
                    batch_X = train_X[pos:pos+batch_size]
                    batch_y = train_y[pos:pos+batch_size]
                    batch_w = train_w[pos:pos+batch_size]

                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)

                    print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
                    train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})

    weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    print("Valid Weighted Classification Accuracy: %f" % weighted_score)

    return weighted_score


# Specify the lists of hyperparameter values to try
n_hidden_values = [50, 100, 200]
n_layers_values = [1, 2, 3]
learning_rate_values = [0.001, 0.01]
dropout_prob_values = [0.5, 0.7]
n_epochs_values = [45, 50, 55]
batch_size_values = [100, 200]
weight_positives_values = [True, False]
num_repeats = 3

# Evaluate different hyperparameter combinations
eval_tox21_hyperparams(n_hidden_values, n_layers_values, learning_rate_values, dropout_prob_values,
                       n_epochs_values, batch_size_values, weight_positives_values)
