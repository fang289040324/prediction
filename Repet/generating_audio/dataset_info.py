import os, pickle
from os.path import join

if __name__ == '__main__':
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    bg_fg = ['background', 'foreground']
    bss_measures = ['sdr', 'sir', 'sar']

    bss_vals = { (b, s): [] for b in bg_fg for s in bss_measures}
    for pick in pickle_folders_to_load:
        sdrs_name = join(pickle_folder, pick, pick + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        [bss_vals[(b, s)].append(sdr_vals[b][s]) for b in bg_fg for s in bss_measures]

    for b in bg_fg:
        for s in bss_measures:
            print b, s, 'min = ', min(bss_vals[(b, s)]), 'max = ', max(bss_vals[(b, s)])
