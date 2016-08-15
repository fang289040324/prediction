# coding=utf-8
"""
Plotting all TRUE sdr values
"""

import os
from os.path import join, isfile, splitext, basename
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def main():
    """

    :return:
    """
    fg_or_bg = 'background'
    sdr_type = 'sdr'

    repet_pickle_folder = 'Repet/pickles_rolloff'
    repet_pickle_folders_to_load = [f for f in os.listdir(repet_pickle_folder)
                                    if os.path.isdir(join(repet_pickle_folder, f))]
    repet_dict = {}
    for folder in repet_pickle_folders_to_load:
        sdrs_path = join(repet_pickle_folder, folder, folder + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_path, 'rb'))
        name = splitext(basename(sdrs_path))[0]
        name = name[:-4]
        repet_dict[name] = sdr_vals[fg_or_bg][sdr_type]
    # repet_dict = sorted(repet_dict)

    nmf_pickle_folder = 'NMF/MFCC/mfcc_pickles'
    nmf_pickles_to_load = [join(nmf_pickle_folder, f) for f in os.listdir(nmf_pickle_folder)
                           if isfile(join(nmf_pickle_folder, f))
                           and splitext(join(nmf_pickle_folder, f))[1] == '.pick']

    nmf_dict = {}
    for pickle_file in nmf_pickles_to_load:
        pickle_dict = pickle.load(open(pickle_file, 'rb'))
        name = splitext(basename(pickle_dict))[0]
        nmf_dict[name] = pickle_dict['sdr_dict'][fg_or_bg][sdr_type]
    # nmf_dict = sorted(nmf_dict)

    repet = []
    nmf = []
    best = []
    assert len(repet_dict) == len(nmf_dict)
    for name in repet_dict:
        repet.append(repet_dict[name])
        nmf.append(nmf_dict[name])
        best.append(np.max((repet_dict[name], nmf_dict[name])))

    repet = np.array(repet)
    nmf = np.array(nmf)
    best = np.array(best)

    plt.plot(repet, nmf)
    plt.xlabel('REPET SDR')
    plt.ylabel('NMF MFCC SDR')
    plt.savefig('repet_nmf.png')


if __name__ == '__main__':
    main()
