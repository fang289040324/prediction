"""
cool stuff here
"""

from __future__ import division, print_function, absolute_import

import pickle
import os
import numpy as np
from os.path import join

import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def main():
    """

    :return:
    """
    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'sim_mat'
    beat_spec_len = 432

    # set up training, testing, & validation partitions
    sim_mat_array, sdr_array = get_generated_data(feature, fg_or_bg, sdr_type)

    # training params
    n_classes = 10
    n_training_steps = 1000
    training_step_size = 100
    training_percent = 0.85
    testing_percent = 0.15
    validation_percent = 0.00

    sdr_array_1h, hist = sdrs_to_one_hots(sdr_array, n_classes, True)

    train, test, validate = split_into_sets(len(sim_mat_array), training_percent,
                                            testing_percent, validation_percent)

    # Building convolutional network
    network = input_data(shape=[None, beat_spec_len, 1], name='input')
    network = conv_1d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    # network = local_response_normalization(network)
    # network = batch_normalization(network)
    # network = conv_1d(network, 64, 3, activation='relu', regularizer="L2")
    # network = max_pool_1d(network, 2)
    # network = local_response_normalization(network)
    # network = batch_normalization(network)
    # network = fully_connected(network, 128, activation='tanh')
    # network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, n_classes, activation='softmax')
    # network = fully_connected(network, 1, activation='linear')
    network = regression(network, optimizer='adagrad', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    X = np.expand_dims(sim_mat_array, -1)
    Y = np.array(sdr_array_1h)
    # X = np.expand_dims([beat_spec_array[i] for i in train], -1)
    # Y = np.array([sdr_array_1h[i] for i in train])
    # testX = np.expand_dims([beat_spec_array[i] for i in test], -1)
    # testY = np.array([sdr_array[i] for i in test])

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=1)
    model.fit({'input': X}, {'target': Y}, n_epoch=20,
              validation_set=0.1,
              snapshot_step=1000, show_metric=True, run_id='{} classes'.format(n_classes - 1))

    # for i in range(len(test)):
    #     print('prediction = ', model.predict(testX[i]), ' actual = ', testY[i])


def split_into_sets(length, training_percent, testing_percent, validation_percent):
    """
    Returns indices to loop through
    :param dataset:
    :return:
    """
    assert (training_percent + testing_percent + validation_percent) == 1.0

    perm = np.random.permutation(length) # random permutation of indices
    train_i = int(len(perm) * training_percent)
    test_i = train_i + int(len(perm) * testing_percent)
    validate_i = test_i + int(len(perm) * validation_percent)

    train = perm[:train_i]
    test = perm[train_i:test_i]
    validate = perm[test_i:]

    return train, test, validate


def sdrs_to_one_hots(sdr_array, n_classes, verbose):
    """
    splits an sdr_array into n_classes and outputs as a one_hot array (2D)
    :param sdr_array:
    :param n_classes:
    :param verbose:
    :return:
    """
    sdr_array = np.array(sdr_array)

    hist = np.histogram(sdr_array, bins=n_classes - 1)
    diff = hist[1][1] - hist[1][0]

    if verbose:
        print(hist[1])
        print('granularity = ', diff)

    sdr_in_bins = np.array((sdr_array - sdr_array.min()) / diff, dtype=int)

    one_hot = np.zeros((sdr_array.size, n_classes))
    one_hot[np.arange(sdr_array.size), sdr_in_bins] = 1
    return one_hot, hist


def get_generated_data(feature, fg_or_bg, sdr_type):
    """
    gets the generated data
    :param feature:
    :param fg_or_bg:
    :param sdr_type:
    :return:
    """
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    return load_feature_and_sdrs(pickle_folders_to_load, pickle_folder, feature, fg_or_bg, sdr_type)


def load_feature_and_sdrs(pickle_folders_to_load, pickle_folder, feature, fg_or_bg, sdr_type):
    """

    :param pickle_folders_to_load:
    :param pickle_folder:
    :param feature:
    :param fg_or_bg:
    :param sdr_type:
    :return:
    """
    beat_spec_array = []
    sdr_array = []

    for folder in pickle_folders_to_load:
        beat_spec_name = join(pickle_folder, folder, folder + '__' + feature + '.pick')
        beat_spec_array.append(pickle.load(open(beat_spec_name, 'rb')))

        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        sdr_array.append(sdr_vals[fg_or_bg][sdr_type])

    return np.array(beat_spec_array), np.array(sdr_array)


if __name__ == '__main__':
    main()
