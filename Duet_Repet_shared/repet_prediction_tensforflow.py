from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pickle
import os
from os.path import join, splitext, isfile
import time

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

def main():
    pickle_folder = 'pickles_combined'


    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'beat_spec'

    # training params
    training_percent = 0.85
    testing_percent = 0.15
    validation_percent = 0.00
    beat_spec_max = 355


    # set up training, testing, & validation partitions
    beat_spec_array, sdr_array = unpickle_beat_spec_and_sdrs(pickle_folder, beat_spec_max)

    train, test, validate = split_into_sets(len(beat_spec_array), training_percent,
                                            testing_percent, validation_percent)

    trainX = np.expand_dims([beat_spec_array[i] for i in train], -1)
    trainY = np.expand_dims([sdr_array[i] for i in train], -1)
    testX = np.expand_dims([beat_spec_array[i] for i in test], -1)
    testY = np.array([sdr_array[i] for i in test])

    # Building convolutional network
    network = input_data(shape=[None, beat_spec_max, 1])
    network = conv_1d(network, 32, 4, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = conv_1d(network, 64, 80, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu') # look for non-tanh things???
    network = dropout(network, 0.8)
    network = fully_connected(network, 1, activation='linear')
    regress = tflearn.regression(network, optimizer='sgd', loss='mean_square', learning_rate=0.01)

    start = time.time()
    # Training
    model = tflearn.DNN(regress, tensorboard_verbose=1)
    model.fit(trainX, trainY, n_epoch=2000,
              snapshot_step=1000, show_metric=True, run_id='mir1k_2000_truncate')
    elapsed = (time.time() - start)

    predicted = np.array(model.predict(testX))[:,0]
    print("Test MSE: ", np.square(testY - predicted).mean())
    print(elapsed, "seconds")
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
    plt.savefig('scatter_true_pred_mir1kaug_cnn_complex1_sgd_lr0.01_win4_2000epoch_last_two_relu_truncate.png')


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


def unpickle_beat_spec_and_sdrs(base_folder, max_len):
    list_of_all_pickles = [join(base_folder, f) for f in os.listdir(base_folder) if isfile(join(base_folder, f))
                            and splitext(join(base_folder, f))[1] == '.pick']
    list_of_all_pickles = sorted(list_of_all_pickles)

    max_bs_len = -1
    beat_spectra = np.zeros((len(list_of_all_pickles), max_len))
    sdrs = np.zeros((len(list_of_all_pickles)))
    i = 0
    for pick in list_of_all_pickles:
        pick_dict = pickle.load(open(pick, 'rb'))
        max_bs_len = len(pick_dict['beat_spec']) if len(pick_dict['beat_spec']) > max_bs_len else max_bs_len
        beat_spectra[i, :] = pick_dict['beat_spec'][:max_len]
        sdrs[i] = pick_dict['repet_sdr_dict']['background']['sdr']
        i += 1

    # beat_spectra = np.array([np.pad(bs, (0, max_bs_len - len(bs)), 'constant', constant_values=(0,))
    #                          for bs in beat_spectra])

    return np.array(beat_spectra), np.array(sdrs)


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


if __name__ == '__main__':
    main()
