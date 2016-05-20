# coding=utf-8
"""
Predicting original REPET SDR values using similarity matrices from the same files
"""

from __future__ import division, print_function, absolute_import

import pickle
import os
import numpy as np
from os.path import join

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def main():
    """
    the main party
    :return:
    """
    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'sim_mat'
    l = 432
    n_classes = 11
    testing_percent = 0.1

    sim_mat_array, sdr_array = get_generated_data(feature, fg_or_bg, sdr_type)
    sim_mat_array = np.expand_dims(sim_mat_array, -1)

    sdr_array_1h, hist = sdrs_to_one_hots(sdr_array, n_classes, True)

    network = input_data(shape=[None, l, l, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, n_classes, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': sim_mat_array}, {'target': sdr_array_1h}, n_epoch=10,
              validation_set=testing_percent,
              snapshot_step=100, show_metric=True, run_id='convnet_mnist')

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

