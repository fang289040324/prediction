import os, pickle
from os.path import join

if __name__ == '__main__':
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    bg_fg = ['background', 'foreground']
    bss_measures = ['sdr', 'sir', 'sar']

    bss_vals = { (b, s): [] for b in bg_fg for s in bss_measures}
    all_bss = []
    for pick in pickle_folders_to_load:
        sdrs_name = join(pickle_folder, pick, pick + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        # [bss_vals[(b, s)].append(sdr_vals[b][s]) for b in bg_fg for s in bss_measures]
        for b in bg_fg:
            for s in bss_measures:
                all_bss.append((sdr_vals[b][s], os.path.split(pick)[1], b, s))

    avg = sum(a[0] for a in all_bss) / len(all_bss)
    # all_bss.sort(key=lambda tup: tup[0])
    #
    # f = open('bss_sorted.txt', 'w')
    # for bss in all_bss:
    #     f.write('{0:+2f} \t {1:40} \t {2:16} \t {3:5}\n'.format(*bss))
