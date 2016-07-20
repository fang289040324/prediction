from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pickle
import os
from os.path import join

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

def main():
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'beat_spec'
    beat_spec_len = 432
    n_statistics = 2

    # training params
    training_percent = 0.85
    testing_percent = 0.15
    validation_percent = 0.00


    # set up training, testing, & validation partitions
    beat_spec_array, sdr_array = load_beat_spec_and_sdrs(pickle_folders_to_load, pickle_folder,
                                                         feature, fg_or_bg, sdr_type)

    train, test, validate = split_into_sets(len(pickle_folders_to_load), training_percent,
                                            testing_percent, validation_percent)

    trainX = np.expand_dims([beat_spec_array[i] for i in train], -1)
    trainY = np.expand_dims([sdr_array[i] for i in train], -1)
    testX = np.expand_dims([beat_spec_array[i] for i in test], -1)
    testY = np.array([sdr_array[i] for i in test])

    # Building convolutional network
    network = input_data(shape=[None, n_statistics, 1])
    network = conv_1d(network, 32, 40, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = conv_1d(network, 64, 80, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 1, activation='linear')
    regress = tflearn.regression(network, optimizer='adagrad', loss='mean_square', learning_rate=0.001)

    # Training
    model = tflearn.DNN(regress, tensorboard_verbose=1)
    model.fit(trainX, trainY, n_epoch=100,
              snapshot_step=1000, show_metric=True, run_id='beat_spec_statistics_1')

    predicted = np.array(model.predict(testX))[:,0]
    print("Test MSE: ", np.square(testY - predicted).mean())
    plot(testY, predicted)


def plot(true, pred):
    """

    :param true:
    :param pred:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.plot(true, pred, '.', color='blue', label='data')
    plt.plot(np.linspace(-50.0, 50.0),np.linspace(-50.0, 50.0), '--', color='red', label='y=x')
    plt.plot(np.linspace(-50.0, 50.0),np.linspace(-45.0, 55.0), '--', color='green', label='+/- 5dB')
    plt.plot(np.linspace(-50.0, 50.0),np.linspace(-55.0, 45.0), '--', color='green')
    plt.title('Generated data')
    plt.xlabel('True SDR (dB)')
    plt.xlim(-15, 40)
    plt.ylim(-15, 40)
    plt.ylabel('Predicted SDR (dB)')
    plt.legend(loc='lower right')
    # plt.colorbar()
    plt.savefig('scatter_true_pred_cnn_complex1_adagrad_lr0.001_win40_100epoch.png')

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

def beat_spectrum_prediction_statistics(beat_spectrum):
    beat_spec_norm = beat_spectrum / np.max(beat_spectrum)

    entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    log_mean = np.log(np.mean(beat_spectrum[1:]))
    # beat_spectrum = beat_spectrum[:1]
    # beat_spec_norm = beat_spectrum / np.max(beat_spectrum)
    #
    # entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    # log_mean = np.log(np.mean(beat_spectrum))

    return entropy, log_mean


if __name__ == '__main__':
    main()
