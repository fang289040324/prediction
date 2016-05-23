from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pickle
import os
from os.path import join

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

def main():
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'sim_mat'
    beat_spec_len = 432

    # training params
    n_classes = 16
    training_percent = 0.85
    testing_percent = 0.15
    validation_percent = 0.00


    # set up training, testing, & validation partitions
    print('Loading sim_mat and sdrs')
    sim_mat_array, sdr_array = get_generated_data(feature, fg_or_bg, sdr_type)
    print('sim_mat and sdrs loaded')

    print('splitting and grooming data')
    train, test, validate = split_into_sets(len(pickle_folders_to_load), training_percent,
                                            testing_percent, validation_percent)

    trainX = np.expand_dims([sim_mat_array[i] for i in train], -1)
    trainY = np.expand_dims([sdr_array[i] for i in train], -1)
    testX = np.expand_dims([sim_mat_array[i] for i in test], -1)
    testY = np.array([sdr_array[i] for i in test])

    print('setting up CNN')
    # Building convolutional network
    network = input_data(shape=[None, beat_spec_len, beat_spec_len, 1])
    network = conv_2d(network, 32, 10, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 20, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 1, activation='linear')
    regress = tflearn.regression(network, optimizer='sgd', loss='mean_square', learning_rate=0.01)

    print('running CNN')
    # Training
    model = tflearn.DNN(regress, tensorboard_verbose=1)
    model.fit(trainX, trainY, n_epoch=10,
              snapshot_step=1000, show_metric=True, run_id='{} classes'.format(n_classes - 1))

    predicted = np.array(model.predict(testX))[:,0]

    print('plotting')
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
    plt.savefig('scatter_true_pred_cnn_sim_mat_rep_org_complex1_10epoch.png')

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
    total = len(pickle_folders_to_load)
    pc = 5
    step = total // (100 // pc)

    i = 0
    for folder in pickle_folders_to_load:
        beat_spec_name = join(pickle_folder, folder, folder + '__' + feature + '.pick')
        beat_spec_array.append(pickle.load(open(beat_spec_name, 'rb')))

        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        sdr_array.append(sdr_vals[fg_or_bg][sdr_type])

        if i % step == 0: print('{}% loaded'.format(float(i)/total * 100))
        i += 1

    return np.array(beat_spec_array), np.array(sdr_array)


if __name__ == '__main__':
    main()
