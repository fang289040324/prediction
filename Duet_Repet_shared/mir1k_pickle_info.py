"""
Get data from MIR-1K pickles
"""

import pickle
import os
from os.path import isfile, join, splitext
import matplotlib.pyplot as plt
import pprint

def main():
    unpickle('pickles_combined')


def unpickle(base_folder):
    list_of_all_pickles = [join(base_folder, f) for f in os.listdir(base_folder) if isfile(join(base_folder, f))
                            and splitext(join(base_folder, f))[1] == '.pick']

    bg_fg = ['background', 'foreground']
    bss_measures = ['sdr', 'sir', 'sar']
    repet_duet = ['repet_sdr_dict', 'duet_sdr_dict']

    # beat_spectra = []
    # repet_fore_sdrs = [pickle.load(open(p, 'rb'))['repet_sdr_dict']['foreground']['sdr'] for p in list_of_all_pickles]
    # repet_back_sdrs = [pickle.load(open(p, 'rb'))['repet_sdr_dict']['background']['sdr'] for p in list_of_all_pickles]
    # duet_fore_sdrs = [pickle.load(open(p, 'rb'))['duet_sdr_dict']['foreground']['sdr'] for p in list_of_all_pickles]
    # duet_back_sdrs = [pickle.load(open(p, 'rb'))['duet_sdr_dict']['background']['sdr'] for p in list_of_all_pickles]
    # avg = [(duet_fore_sdrs[i] + duet_back_sdrs[i]) / 2 for i in range(len(duet_back_sdrs))]
    #
    # print('repet_fore: min ', min(repet_fore_sdrs), ' max ', max(repet_fore_sdrs))
    # print('repet_back: min ', min(repet_back_sdrs), ' max ', max(repet_back_sdrs))
    # print('duet_fore: min ', min(duet_fore_sdrs), ' max ', max(duet_fore_sdrs))
    # print('duet_back: min ', min(duet_back_sdrs), ' max ', max(duet_back_sdrs))
    # print('avg: min ', min(avg), ' max ', max(avg))

    all_bss = []
    for pick in list_of_all_pickles:
        pick_dict = pickle.load(open(pick, 'rb'))
        for b in bg_fg:
            for s in bss_measures:
                for a in repet_duet:
                    all_bss.append((pick_dict[a][b][s], os.path.split(pick)[1], b, s, a))

    all_bss.sort(key=lambda tup: tup[0])

    f = open('bss_sorted.txt', 'w')
    for bss in all_bss:
        f.write('{0:+2f} \t {1:20} \t {2:16} \t {3:5} \t {4:20} \n'.format(*bss))


if __name__ == '__main__':
    main()