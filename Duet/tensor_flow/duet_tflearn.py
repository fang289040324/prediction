from __future__ import division, print_function, absolute_import

import csv
import numpy as np
import os
import pickle

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.metrics import Top_k


def main():
    data = load_data('reverb_pan_full_sdr.txt', 'pickle/')

    n_classes = 20
    test_percent = 0.15

    train, test, validate = split_into_sets(len(data['sdr']), 1-test_percent,
                                            test_percent, 0)

    x_train = np.expand_dims([data['input'][i] for i in train], -1)
    y_train = np.expand_dims([data['sdr'][i] for i in train], -1)
    x_test = np.expand_dims([data['input'][i] for i in test], -1)
    y_test = np.expand_dims([data['sdr'][i] for i in test], -1)

    # X = np.expand_dims(np.array(data['input']), -1)
    # Y =np.array(data['sdr'])
    #
    # y = np.expand_dims(np.array(data['sdr']), -1)
    # Y_1h, hist = np.array(sdrs_to_one_hots(Y, n_classes, True))

    inp = input_data(shape=[None, 50, 50, 1], name='input')
    conv1 = conv_2d(inp, 32, [5, 5], activation='relu', regularizer="L2")
    max_pool = max_pool_2d(conv1, 2)
    conv2 = conv_2d(max_pool, 64, [5, 5], activation='relu', regularizer="L2")
    max_pool2 = max_pool_2d(conv2, 2)
    full = fully_connected(max_pool2, 128, activation='tanh')
    full = dropout(full, 0.8)
    full2 = fully_connected(full, 256, activation='tanh')
    full2 = dropout(full2, 0.8)
    out = fully_connected(full2, 1, activation='linear')
    network = regression(out, optimizer='sgd', learning_rate=0.01, name='target', loss='mean_square')

    model = tflearn.DNN(network, tensorboard_verbose=1, checkpoint_path='checkpoint.p',
                        tensorboard_dir='tmp/tflearn_logs/')

    model.fit({'input': x_train}, {'target': y_train}, n_epoch=1000, validation_set=(x_test, y_test),
              snapshot_step=10000, run_id='convnet_duet_2')

    # for i in xrange(0, 10):
    #     model.fit({'input': x_train}, {'target': y_train}, n_epoch=1, validation_set=(x_test, y_test),
    #               snapshot_step=10000, run_id='convnet_duet')
    #
    predicted = np.array(model.predict(x_test))[:,0]
    #     print("Test MSE epoch ", i, ": ", np.square(y_test - predicted).mean())
    plot(y_test, predicted, 'epoch_'+str(i))


def plot(true, pred, fname):
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
    plt.ylabel('Predicted SDR (dB)')
    plt.legend(loc='lower right')
    # plt.colorbar()
    plt.show()
    plt.savefig(fname + '.pdf')


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
