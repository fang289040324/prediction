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
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.estimator import regression

def main():
    """

    :return:
    """
    pickle_folder = '../Repet/pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]

    fg_or_bg = 'background'
    sdr_type = 'sdr'
    feature = 'beat_spec'
    beat_spec_len = 432
    n_epochs = 200
    take = 1

    # set up training, testing, & validation partitions
    beat_spec_array, sdr_array = load_beat_spec_and_sdrs(pickle_folders_to_load, pickle_folder,
                                                         feature, fg_or_bg, sdr_type)

    beat_spec_array = np.expand_dims(beat_spec_array, -1)
    sdr_array = np.expand_dims(sdr_array, -1)

    # Building convolutional network
    network = input_data(shape=[None, beat_spec_len, 1])
    network = conv_1d(network, 32, 4, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = conv_1d(network, 64, 80, activation='relu', regularizer="L2")
    network = max_pool_1d(network, 2)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')  # look for non-tanh things???
    network = dropout(network, 0.8)
    network = fully_connected(network, 1, activation='linear')
    regress = tflearn.regression(network, optimizer='rmsprop', loss='mean_square', learning_rate=0.001)

    start = time.time()
    # Training
    model = tflearn.DNN(regress)  # , session=sess)
    model.fit(beat_spec_array, sdr_array, n_epoch=n_epochs,
              snapshot_step=1000, show_metric=True,
              run_id='repet_save_{0}_epochs_take_{1}'.format(n_epochs, take))
    elapsed = (time.time() - start)
    print('Finished training after ' + elapsed + 'seconds. Saving...')

    model_output_folder = 'network_outputs/'
    model_output_file = join(model_output_folder, 'repet_save_{0}_epochs_take_{1}'.format(n_epochs, take))

    model.save(model_output_file)


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