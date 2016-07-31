# coding=utf-8
"""
Predicting NMF MFCCs using tensorflow
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pickle
import os
from os.path import join, splitext, isfile
import time
import pprint

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d

def main():
    """

    :return:
    """
    pickle_folder = '../mfcc_pickles'
    pickle_folders_to_load = [join(pickle_folder, f) for f in os.listdir(pickle_folder)
                              if isfile(join(pickle_folder, f))
                              and splitext(join(pickle_folder, f))[1] == '.pick']
    pickle_folders_to_load = sorted(pickle_folders_to_load)
    length = len(pickle_folders_to_load)

    # pickle parameters
    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'mfcc_clusters'
    n_folds = 2
    n_epochs = 20
    take = 1

    output_folder = 'cross_{0}_folds_{1}_epochs_take{2}/'.format(n_folds, n_epochs, take)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summary_file = output_folder + 'summary_{0}_folds_{1}_epochs.txt'.format(n_folds, n_epochs)

    mfcc_array, sdr_array = load_mfcc_and_sdrs(pickle_folders_to_load, pickle_folder, feature, fg_or_bg, sdr_type)

    perm = np.random.permutation(len(pickle_folders_to_load))  # random permutation of indices
    folds = np.array_split(perm, n_folds)  # splits into folds
    predicted = []

    for fold in range(n_folds):
        train_mfcc = np.expand_dims([mfcc_array[i] for i in range(length) if i not in folds[fold]], -1)
        train_sdr = np.expand_dims([sdr_array[i] for i in range(length) if i not in folds[fold]], -1)
        test_mfcc = np.expand_dims([mfcc_array[i] for i in folds[fold]], -1)
        test_sdr = np.expand_dims([sdr_array[i] for i in folds[fold]], -1)

        with tf.Graph().as_default():
            # Building convolutional network
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

            start = time.time()
            # Training
            model = tflearn.DNN(regress)  # , session=sess)
            model.fit(train_mfcc, train_sdr, n_epoch=n_epochs,
                      snapshot_step=1000, show_metric=True,
                      run_id='relus_{0}_{1}_of_{2}'.format(n_epochs, fold + 1, n_folds))
            elapsed = (time.time() - start)

        prediction = np.array(model.predict(test_mfcc))[:, 0]
        with open(output_folder + 'predictions_fold{}.txt'.format(fold + 1), 'a') as f:
            f.write('Training avg = {} \n \n'.format(np.mean(train_sdr)))
            f.write('Actual \t Predicted \n')
            for i in range(len(prediction)):
                f.write('{0} \t {1} \n'.format(test_sdr[i][0], prediction[i]))
        pprint.pprint(prediction)
        predicted.append(prediction)
        with open(summary_file, 'a') as f:
            mse = np.square(test_sdr - prediction).mean()
            f.write('Fold {0}\t mse = {1} dB \t time = {2} min \n'.format(fold + 1, mse, elapsed / 60.))
        plot(test_sdr, prediction, output_folder + 'scatter_fold{}.png'.format(fold + 1))

        tf.reset_default_graph()

    predicted = np.array(predicted).flatten()
    print("Test MSE: ", np.square(sdr_array - predicted).mean())
    plot(sdr_array, predicted, output_folder + 'scatter_all_folds_{}_epochs.png'.format(n_epochs))


def plot(true, pred, name):
    """

    :param true:
    :param pred:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.close('all')
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
    plt.savefig(name)


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



if __name__ == '__main__':
    main()