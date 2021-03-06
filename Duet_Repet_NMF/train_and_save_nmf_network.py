# coding=utf-8
"""
training and saving repet network
"""

import pickle
import os
from os.path import join
import numpy as np
import time

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

def main():
    """

    :return:
    """
    pickle_folder = '../NMF/mfcc_pickles'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]

    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'mfcc_clusters'
    beat_spec_len = 432
    n_epochs = 200
    take = 1

    # set up training, testing, & validation partitions
    mfcc_array, sdr_array = load_mfcc_and_sdrs(pickle_folders_to_load, pickle_folder,
                                                    feature, fg_or_bg, sdr_type)

    mfcc_array = np.expand_dims(mfcc_array, -1)
    sdr_array = np.expand_dims(sdr_array, -1)

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
    model.fit(mfcc_array, sdr_array, n_epoch=n_epochs,
              snapshot_step=1000, show_metric=True,
              run_id='repet_save_{0}_epochs_take_{1}'.format(n_epochs, take))
    elapsed = (time.time() - start)
    print('Finished training after ' + elapsed + 'seconds. Saving...')

    model_output_folder = 'network_outputs/'
    model_output_file = join(model_output_folder, 'nmf_save_{0}_epochs_take_{1}'.format(n_epochs, take))

    model.save(model_output_file)


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