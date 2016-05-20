import tensorflow as tf
import numpy as np
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt

def main():
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'beat_spec'
    beat_spec_len = 432

    # training params
    n_classes = 11
    n_training_steps = 1000
    training_step_size = 100
    training_percent = 0.85
    testing_percent = 0.10
    validation_percent = 0.05

    assert (training_percent + testing_percent + validation_percent) == 1.0

    # set up training, testing, & validation partitions
    beat_spec_array, sdr_array = load_beat_spec_and_sdrs(pickle_folders_to_load, pickle_folder,
                                                         feature, fg_or_bg, sdr_type)

    sdr_array = sdrs_to_one_hots(sdr_array, n_classes, True)

    perm = np.random.permutation(len(pickle_folders_to_load)) # random permutation of indices
    train_i = int(len(perm) * training_percent)
    test_i = train_i + int(len(perm) * testing_percent)
    validate_i = test_i + int(len(perm) * validation_percent)

    train = perm[:train_i]
    test = perm[test_i:validate_i]
    validate = perm[validate_i:]

    # y = softmax(Wx + b)
    beat_specs = tf.placeholder(tf.float32, [None, beat_spec_len])
    weights = tf.Variable(tf.zeros([beat_spec_len, n_classes]))
    biases = tf.Variable(tf.zeros([n_classes]))

    # softmax model
    sdr_pred = tf.nn.softmax(tf.matmul(beat_specs, weights) + biases)

    # costs
    sdr_true = tf.placeholder(tf.float32, [None, n_classes])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(sdr_true * tf.log(sdr_pred), reduction_indices=[1]))

    # gradient descent
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # initialize
    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    # train
    for i in range(n_training_steps):
        indices = np.random.choice(train, training_step_size, False)
        beat_spec_train = np.array([beat_spec_array[i] for i in indices])
        sdr_train = np.array([sdr_array[i] for i in indices])
        session.run(train_step, feed_dict={beat_specs: beat_spec_train, sdr_true: sdr_train})

    # test
    correct_prediction = tf.equal(tf.argmax(sdr_pred, 1), tf.argmax(sdr_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # get testing set
    beat_spec_test = np.array([beat_spec_array[i] for i in test])
    sdr_test = np.array([sdr_array[i] for i in test])

    print session.run(accuracy, feed_dict={beat_specs: beat_spec_test, sdr_true: sdr_test})



def load_beat_spec_and_sdrs(pickle_folders_to_load, pickle_folder,
                            feature, fg_or_bg, sdr_type):
    beat_spec_array = []
    sdr_array = []

    for folder in pickle_folders_to_load:
        beat_spec_name = join(pickle_folder, folder, folder + '__' + feature + '.pick')
        beat_spec_array.append(pickle.load(open(beat_spec_name, 'rb')))

        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        sdr_array.append(sdr_vals[fg_or_bg][sdr_type])

    return np.array(beat_spec_array), np.array(sdr_array)

def sdrs_to_one_hots(sdr_array, n_classes, verbose):
    sdr_array = np.array(sdr_array)

    hist = np.histogram(sdr_array, bins=n_classes - 1)
    diff = hist[1][1] - hist[1][0]

    if verbose:
        print hist[1]
        print 'granularity = ', diff

    sdr_in_bins = np.array((sdr_array - sdr_array.min()) / diff, dtype=int)

    one_hot = np.zeros((sdr_array.size, n_classes))
    one_hot[np.arange(sdr_array.size), sdr_in_bins] = 1
    return one_hot


if __name__ == '__main__':
    main()
