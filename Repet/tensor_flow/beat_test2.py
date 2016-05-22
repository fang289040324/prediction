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

    # training params
    n_classes = 16
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
    input = input_data(shape=[None, beat_spec_len, 1])
    conv1 = conv_1d(input, 32, 10, activation='relu', regularizer="L2")
    max_pool1 = max_pool_1d(conv1, 2)
    full = fully_connected(max_pool1, 512, activation='tanh')
    # single = tflearn.single_unit(full)
    single = fully_connected(full, 1, activation='linear')
    regress = tflearn.regression(single, optimizer='sgd', loss='mean_square', learning_rate=0.01)

    # Training
    model = tflearn.DNN(regress, tensorboard_verbose=1)
    model.fit(trainX, trainY, n_epoch=100,
              snapshot_step=1000, show_metric=True, run_id='{} classes'.format(n_classes - 1))

    predicted = np.array(model.predict(testX))[:,0]
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
    plt.savefig('scatter_true_pred_cnn_simple_100epoch.png')

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

def sdrs_to_one_hots(sdr_array, n_classes, verbose):
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


if __name__ == '__main__':
    main()
