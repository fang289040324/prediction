import os
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.fftpack
import operator


def main():
    out = '../analysis_output/'
    pickle_folder = '../pickles'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    sdr_dict = {}
    coefs = {}
    sdrs = []

    exclude = ['noise_', 'speed', '0.0']
    include = ['noise_', '0.0']


    noise_time = 'noise_time'
    speed = 'speed'
    stft_swap = 'stft_swap'

    sdr_by_noise_type = {noise_time: {}, speed: {}, stft_swap: {}}
    coefs_by_noise_type = {noise_time: set(), speed: set(), stft_swap: set()}

    sdr_type = 'foreground'

    perfect_coef = 0.0
    first = True


    for folder in pickle_folders_to_load:
        beat_spec_name = join(pickle_folder, folder, folder + '__beat_spec.pick')
        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')

        # beat_spec = pickle.load(open(beat_spec_name, 'rb'))
        # # dct = scipy.fftpack.dct(beat_spec)
        # if any([folder.find(e) != -1 for e in exclude]):
        #     continue

        coef = float(folder.split('_')[-1])
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        cur_name = folder.split('_')[0] + '__' + folder.split('_')[2]

        if coef == perfect_coef:
            if cur_name not in sdr_by_noise_type[noise_time]:
                sdr_by_noise_type[noise_time][cur_name] = []
            if cur_name not in sdr_by_noise_type[speed]:
                sdr_by_noise_type[speed][cur_name] = []
            if cur_name not in sdr_by_noise_type[stft_swap]:
                sdr_by_noise_type[stft_swap][cur_name] = []

            perfect_sdr = sdr_vals[sdr_type][0]

            # sdr_by_noise_type[noise_time][cur_name].append(sdr_vals[sdr_type][0] - perfect_sdr)
            # sdr_by_noise_type[speed][cur_name].append(sdr_vals[sdr_type][0] - perfect_sdr)
            # sdr_by_noise_type[stft_swap][cur_name].append(sdr_vals[sdr_type][0] - perfect_sdr)

            continue


        if folder.find(noise_time) != -1:
            sdr_by_noise_type[noise_time][cur_name].append(sdr_vals[sdr_type][0] - perfect_sdr)
            coefs_by_noise_type[noise_time].add(coef)

        elif folder.find(speed) != -1:
            sdr_by_noise_type[speed][cur_name].append(sdr_vals[sdr_type][0] - perfect_sdr)
            coefs_by_noise_type[speed].add(coef)

        else:
            sdr_by_noise_type[stft_swap][cur_name].append(sdr_vals[sdr_type][0] - perfect_sdr)
            coefs_by_noise_type[stft_swap].add(coef)


    for noise_type, coefs in sdr_by_noise_type.iteritems():
        # sorted_sdrs = sorted(coefs.items(), key=operator.itemgetter(0))
        sdr_values = zip(*[diffs for diffs in coefs.values()])
        sdr_coefs = sorted(coefs_by_noise_type[noise_type])
        sdr_coefs.insert(0, '')

        plt.close('all')
        # plt.yscale('log')
        plt.violinplot(sdr_values, showmeans=False, showmedians=True)

        plt.xticks(range(len(sdr_values)+1), sdr_coefs, rotation='vertical')
        plt.title('repetition perturbation coef vs. foreground SDR {}'.format(noise_type))
        plt.xlabel('perturbation coefficient')
        plt.ylabel('SDR (dB)')
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(out + 'bigger_dataset_sdr_foreground_diff_{}.png'.format(noise_type))





if __name__ == '__main__':
    main()
