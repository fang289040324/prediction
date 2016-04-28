import nussl
import os
from os.path import join, splitext
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    pickle_folder = '../mir_1k/pickles_random'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if '__beat_spec.pick' in f]
    pickle_folders_to_load = sorted(pickle_folders_to_load)
    out = '../mir_1k/beat_spec_random/'

    smallest = (100, '')
    biggest = (-100, '')
    for pick in pickle_folders_to_load:
        bs_path = join(pickle_folder, pick)
        beat_spec = pickle.load(open(bs_path, 'rb'))

        pick = pick.replace('__beat_spec.pick', '')

        sdrs_name = join(pickle_folder, pick + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        sdr = sdr_vals['background']['sdr']
        cur_sdr = '%.2f' % sdr

        if sdr < smallest[0]:
            smallest = (sdr, pick)
        if sdr > biggest[0]:
            biggest = (sdr, pick)

        out_name = join(out, pick + '.png')

        # plt.close('all')
        # plt.plot(beat_spec)
        # plt.title(pick + '.wav    sdr=' + cur_sdr)
        # plt.savefig(out_name)
        # print 'wrote ' + out_name

    print smallest
    print biggest
