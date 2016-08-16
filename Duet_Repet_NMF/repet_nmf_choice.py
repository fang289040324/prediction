# coding=utf-8
"""
repet nmf choice
"""

import pickle
import os
from os.path import join, isfile, splitext
import numpy as np
import time

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression


def main():
    """

    :return:
    """
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    repet_feature = 'beat_spec'
    nmf_feature = 'mfcc_clusters'
    n_epochs_repet = 100
    n_epochs_nmf = 20
    take = 9
    test_percent = 0.1
    output_file = 'summary_repet_nmf_choice_take{}.txt'.format(take)

    repet_pickle_folder = '../Repet/pickles_rolloff'
    repet_pickle_folders_to_load = [f for f in os.listdir(repet_pickle_folder)
                                    if os.path.isdir(join(repet_pickle_folder, f))]
    repet_pickle_folders_to_load = sorted(repet_pickle_folders_to_load)
    print(str(len(repet_pickle_folders_to_load)))

    beat_spec_array, repet_sdr_array = load_beat_spec_and_sdrs(repet_pickle_folders_to_load, repet_pickle_folder,
                                                               repet_feature, fg_or_bg, sdr_type)

    nmf_pickle_folder = '../NMF/mfcc_pickles'
    nmf_pickles_to_load = [join(nmf_pickle_folder, f) for f in os.listdir(nmf_pickle_folder)
                           if isfile(join(nmf_pickle_folder, f))
                           and splitext(join(nmf_pickle_folder, f))[1] == '.pick']
    nmf_pickles_to_load = sorted(nmf_pickles_to_load)

    mfcc_array, mfcc_sdr_array = load_mfcc_and_sdrs(nmf_pickles_to_load, nmf_pickle_folder,
                                                    nmf_feature, fg_or_bg, sdr_type)

    print(beat_spec_array.shape, mfcc_array.shape)

    train_ind, test_ind = split_into_sets(len(nmf_pickles_to_load), test_percent)

    repet_trainX = np.expand_dims([beat_spec_array[i] for i in train_ind], -1)
    repet_trainY = np.expand_dims([repet_sdr_array[i] for i in train_ind], -1)
    repet_testX = np.expand_dims([beat_spec_array[i] for i in test_ind], -1)
    repet_testY = np.array([repet_sdr_array[i] for i in test_ind])

    nmf_trainX = np.expand_dims([mfcc_array[i] for i in train_ind], -1)
    nmf_trainY = np.expand_dims([mfcc_sdr_array[i] for i in train_ind], -1)
    nmf_testX = np.expand_dims([mfcc_array[i] for i in test_ind], -1)
    nmf_testY = np.array([mfcc_sdr_array[i] for i in test_ind])

    repet_network = train_repet_network(repet_trainX, repet_trainY, n_epochs_repet, take)
    nmf_network = train_nmf_network(nmf_trainX, nmf_trainY, n_epochs_nmf, take)

    repet_predictions = np.array(repet_network.predict(repet_testX))[:, 0]
    nmf_predictions = np.array(nmf_network.predict(nmf_testX))[:, 0]

    prediction_results = []
    diffs = []
    for i in range(len(test_ind)):
        repet_predicted = repet_predictions[i] > nmf_predictions[i]
        repet_actual = repet_testY[i] > nmf_testY[i]
        prediction_results.append(repet_predicted == repet_actual)
	if repet_predicted != repet_actual:
	    diffs.append(np.abs(repet_predictions[i] - nmf_predictions[i]))

    diffs = np.array(diffs)
    percent_correct = float(sum(1 for p in prediction_results if p)) / float(len(prediction_results))
    
    with open(output_file, 'w') as f:
	f.write('source: ' + fg_or_bg + '\nquality type: ' + sdr_type + '\n# repet epochs: ' + str(n_epochs_repet))
	f.write('\n# nmf epochs: ' + str(n_epochs_nmf) + '\ntest percent: ' + str(test_percent * 10) + '%\n')
	f.write('\n# test examples: ' + str(len(test_ind)) + '\nrepet loss: ' + str(repet_network.train_ops[0].loss_value))
	f.write('\nnmf loss: ' + str(nmf_network.train_ops[0].loss_value) + '\n')
        f.write('percent predictions correct: ' + str(percent_correct * 100) + '%\n')
	f.write('\naverage difference of incorrect: ' + str(np.mean(diffs)) + '+/- ' + str(np.std(diffs)) + 'dBs\n')
    print(percent_correct)


def train_repet_network(beat_spectrum_array, sdr_array, n_epochs, take):
    """

    :param beat_spectrum_array:
    :param sdr_array:
    :param n_epochs:
    :param take:
    :return:
    """
    beat_spec_len = 432
    with tf.Graph().as_default():
        input_layer = input_data(shape=[None, beat_spec_len, 1])
        conv1 = conv_1d(input_layer, 32, 4, activation='relu', regularizer="L2")
        max_pool1 = max_pool_1d(conv1, 2)
        conv2 = conv_1d(max_pool1, 64, 80, activation='relu', regularizer="L2")
        max_pool2 = max_pool_1d(conv2, 2)
        fully1 = fully_connected(max_pool2, 128, activation='relu')
        dropout1 = dropout(fully1, 0.8)
        fully2 = fully_connected(dropout1, 256, activation='relu')
        dropout2 = dropout(fully2, 0.8)
        linear = fully_connected(dropout2, 1, activation='linear')
        regress = tflearn.regression(linear, optimizer='rmsprop', loss='mean_square', learning_rate=0.001)

        # Training
        model = tflearn.DNN(regress)  # , session=sess)
        model.fit(beat_spectrum_array, sdr_array, n_epoch=n_epochs,
                  snapshot_step=1000, show_metric=True,
                  run_id='repet_choice_{0}_epochs_take_{1}'.format(n_epochs, take))

        return model


def train_nmf_network(mfcc_array, sdr_array, n_epochs, take):
    """

    :param mfcc_array:
    :param sdr_array:
    :param n_epochs:
    :param take:
    :return:
    """
    with tf.Graph().as_default():
        network = input_data(shape=[None, 13, 100, 1])
        network = conv_2d(network, 32, [5, 5], activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = conv_2d(network, 64, [5, 5], activation='relu', regularizer="L2")
        network = max_pool_2d(network, 2)
        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 1, activation='linear')
        regress = tflearn.regression(network, optimizer='rmsprop', loss='mean_square', learning_rate=0.001)

        # Training
        model = tflearn.DNN(regress)  # , session=sess)
        model.fit(mfcc_array, sdr_array, n_epoch=n_epochs,
                  snapshot_step=1000, show_metric=True,
                  run_id='repet_choice_{0}_epochs_take_{1}'.format(n_epochs, take))

        return model


def split_into_sets(length, testing_percent):
    """
    Returns indices to loop through
    :param dataset:
    :return:
    """
    training_percent = 1.0 - testing_percent

    perm = np.random.permutation(length) # random permutation of indices
    train_i = int(len(perm) * training_percent)

    train = perm[:train_i]
    test = perm[train_i:]

    return train, test


def load_mfcc_and_sdrs(pickle_file_list, pickle_folder,
                       feature, fg_or_bg, sdr_type):
    """

    :param pickle_file_list:
    :param pickle_folder:
    :param feature:
    :param fg_or_bg:
    :param sdr_type:
    :return:
    """
    mfcc_array = []
    sdr_array = []

    for pickle_file in pickle_file_list:
        pickle_dict = pickle.load(open(pickle_file, 'rb'))
        mfcc_array.append(pickle_dict[feature])
        sdr_array.append(pickle_dict['sdr_dict'][fg_or_bg][sdr_type])

    return np.array(mfcc_array), np.array(sdr_array)


def load_beat_spec_and_sdrs(pickle_folders_to_load, pickle_folder,
                                feature, fg_or_bg, sdr_type):
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
