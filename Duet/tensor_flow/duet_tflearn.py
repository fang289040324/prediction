from __future__ import division, print_function, absolute_import

import csv
import numpy as np
import os
import pickle


def main():
    data = load_data('reverb_pan_full_sdr.txt', 'pickle/')

    # train, test, validate = split_into_sets(len(data['sdr']), training_percent=0.85, testing_percent=0.15, validation_percent=0.0)
    # x_train, x_test, y_train, y_test = np.array(data['input'])[train], np.array(data['input'])[test], np.array(data['sdr'])[train], np.array(data['sdr'])[test]
    #
    # x_train = np.expand_dims([data['input'][i] for i in train], -1)
    # y_train = np.array([data['sdr'][i] for i in train])
    # x_test = np.expand_dims([data['input'][i] for i in test], -1)
    # y_test = np.array([data['sdr'][i] for i in test])

    n_classes = 20
    test_percent = 0.15

    X = np.expand_dims(np.array(data['input']), -1)
    Y =np.array(data['sdr'])

    y = np.expand_dims(np.array(data['sdr']), -1)
    Y_1h, hist = np.array(sdrs_to_one_hots(Y, n_classes, True))


    import tflearn
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.estimator import regression
    from tflearn.layers.normalization import local_response_normalization
    from tflearn.metrics import Top_k

    network = input_data(shape=[None, 50, 50, 1], name='input')
    network = conv_2d(network, 32, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 1, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001, name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='checkpoint.p')
    model.fit({'input': X}, {'target': y}, n_epoch=100,
              validation_set=test_percent,
              snapshot_step=100, show_metric=True, run_id='convnet_duet')


def load_data(sdr_fn, pickle_dir):
    with open(sdr_fn, 'r') as sdr_f:
        sdr_stats = {}
        q2 = csv.DictReader(sdr_f, delimiter='\t',
                            fieldnames=['filename1', 'filename2', 'sdr', 'sir', 'sar', 'perm', 'note'])
        for line2 in q2:
            for entry in line2.keys():
                if entry is not None:
                    if entry not in sdr_stats.keys():
                        sdr_stats[entry] = []
                    if 'name' in entry or 'note' in entry:
                        sdr_stats[entry].append(line2[entry].replace("'", "").replace("]", "").replace("[",""))
                    else:
                        sdr_stats[entry].append(
                            np.fromstring(line2[entry].translate(None, '][()'), sep=' '))

    plot_stats = dict()

    plot_stats['fname'] = []
    plot_stats['sdr'] = []
    plot_stats['input'] = []

    # pick different bc values
    for i in xrange(len(sdr_stats['sdr'])):
        if np.average(sdr_stats['sdr'][i]) < -100:
            continue
        if '-5' in sdr_stats['note'][i] or '-4' in sdr_stats['note'][i]:
            continue

        plot_stats['sdr'].append(np.average(sdr_stats['sdr'][i]))
        notes = sdr_stats['note'][i].split(', ')
        fname1 = os.path.splitext(os.path.split(sdr_stats['filename1'][i])[1])[0]
        fname2 = os.path.splitext(os.path.split(sdr_stats['filename2'][i])[1])[0]
        plot_stats['fname'].append( fname1 + "+" + fname2 + '_' + notes[0]+ "_" + notes[1])

    for j in plot_stats['fname']:
        d = pickle.load(open(pickle_dir+j+".p",'rb'))
        plot_stats['input'].append(d)

    return plot_stats


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

if __name__ == '__main__':
    main()
