import os
from os.path import join
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.fftpack
from mpl_toolkits.mplot3d import Axes3D

def main():
    out = '../analysis_output/beat_spec_all'
    out = '../analysis_output'
    pickle_folder = '../mir_1k/pickles'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if '__beat_spec.pick' in f]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    sdr_type = 'background'
    zero_stft_nr = 'zero_nr_stft'
    swap_stft = 'swap_stft'
    speed = 'speed'
    none = 'none'
    perturbations = [zero_stft_nr, swap_stft, speed, none]

    l = len(pickle_folders_to_load)
    i = 1

    # sdrs = {p: [] for p in perturbations}
    # ents = {p: [] for p in perturbations}
    sdrs = []
    ents = []
    mean = []
    for pick in pickle_folders_to_load:
        pick = pick.replace('__beat_spec.pick', '')

        sdrs_name = join(pickle_folder, pick + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        cur_sdr = sdr_vals[sdr_type][0]
        # sdrs[perturb_type].append(cur_sdr)
        sdrs.append(cur_sdr)

        beat_spec_path = join(pickle_folder, pick + '__beat_spec.pick')
        beat_spec = pickle.load(open(beat_spec_path, 'rb'))


        # dct = scipy.fftpack.dct(beat_spec)
        # dct_norm = np.abs(dct / np.max(dct))
        beat_spec_norm = beat_spec / np.max(beat_spec)
        mean.append(np.log(np.mean(beat_spec[1:])))
        entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
        ents.append(entropy)
        # ents[perturb_type].append(entropy)

        # if perturb_type == swap_stft and cur_sdr <= -1.0 and entropy > 60.0:
        #     print folder
        # return

        # plt.close('all')
        # plt.grid('on')
        # plt.plot(beat_spec, label=folder)
        # plt.legend()
        # plt.title('Entropy: {0:.5f}, SDR: {1:.2f}dB'.format(entropy, cur_sdr))
        # plt.savefig(join(out, '{}.png'.format(folder)))

        # print 'finished ' + folder + ' ' + str(i) + ' of ' + str(l)
        # i += 1

    # for p in perturbations:
    #     plt.close('all')
    #     plt.grid('on')
    #     plt.plot(ents[p], sdrs[p], '.')
    #     plt.xlabel('entropy')
    #     plt.ylabel('sdr (dB)')
    #     plt.title('Entropy of beat spectrum vs. SDRs ({})'.format(p))
    #     plt.savefig(join(out, 'entropy_sdrs_{}.png'.format(p)))


    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ents, mean, sdrs, marker='.')
    ax.set_xlabel('entropy')
    ax.set_ylabel('max')
    ax.set_zlabel('sdr (dB)')

    plt.title('Entropy of beat spectrum norm & max (1st excluded) vs. SDRs')
    plt.savefig(join(out, 'entropy_max_1inc_sdrs_3d_1.png'))










if __name__ == '__main__':
    main()