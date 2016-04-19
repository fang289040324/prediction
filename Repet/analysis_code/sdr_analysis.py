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

    sdr_dict = {}
    coefs = {}
    sdrs = []

    exclude = ['noise_', 'speed', '0.0']
    include = ['noise_', '0.0']


    noise_time = 'noise_time'
    speed = 'speed'
    stft_swap = 'stft_swap'

    sdr_by_noise_type = {noise_time: {}, speed: {}, stft_swap: {}}

    sdr_type = 'background'

    perfect_coef = 0.0
    first = True

    for folder in [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]:
        beat_spec_name = join(pickle_folder, folder, folder + '__beat_spec.pick')
        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')

        name = os.path.basename(folder)
        name = splitext(name)[0]

        # beat_spec = pickle.load(open(beat_spec_name, 'rb'))
        # # dct = scipy.fftpack.dct(beat_spec)
        # if any([folder.find(e) != -1 for e in exclude]):
        #     continue

        coef = float(folder.split('_')[-1])
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))

        if folder.find(noise_time) != -1:
            if coef not in sdr_by_noise_type[noise_time]:
                sdr_by_noise_type[noise_time][coef] = []

            sdr_by_noise_type[noise_time][coef].append(sdr_vals[sdr_type][0])

        elif folder.find(speed) != -1:
            if coef not in sdr_by_noise_type[speed]:
                sdr_by_noise_type[speed][coef] = []

            sdr_by_noise_type[speed][coef].append(sdr_vals[sdr_type][0])

        elif coef != perfect_coef:
            if coef not in sdr_by_noise_type[stft_swap]:
                sdr_by_noise_type[stft_swap][coef] = []

            sdr_by_noise_type[stft_swap][coef].append(sdr_vals[sdr_type][0])

        if coef == perfect_coef:
            if first:
                sdr_by_noise_type[noise_time][coef] = []
                sdr_by_noise_type[speed][coef] = []
                sdr_by_noise_type[stft_swap][coef] = []
                first = False

            sdr_by_noise_type[noise_time][coef].append(sdr_vals[sdr_type][0])
            sdr_by_noise_type[speed][coef].append(sdr_vals[sdr_type][0])
            sdr_by_noise_type[stft_swap][coef].append(sdr_vals[sdr_type][0])


        # if not any([folder.find(i) != -1 for i in include]):
        #     continue


        # sdr_dict[folder] = (sdr_vals[signal][0])

        # if coef not in coefs:
        #     coefs[coef] = []
        #
        # coefs[coef].append(sdr_vals[sdr_type][0])

    for noise_type, coefs in sdr_by_noise_type.iteritems():
        sorted_sdrs = sorted(coefs.items(), key=operator.itemgetter(0))
        sdr_values = [sorted_sdrs[v][1] for v in range(len(sorted_sdrs))]
        sdr_coefs = [str(sorted_sdrs[k][0]) for k in range(len(sorted_sdrs))]
        sdr_coefs.insert(0, '')

        plt.close('all')
        # plt.yscale('log')
        plt.violinplot(sdr_values, showmeans=False, showmedians=True)
        plt.xticks(range(len(sdr_values)+1), sdr_coefs, rotation='vertical')
        plt.title('repetition perturbation coef vs. background SDR {}'.format(noise_type))
        plt.xlabel('perturbation coefficient')
        # plt.xlim([-0.05, 0.5])
        plt.ylabel('SDR (dB)')
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(out + 'bigger_dataset_sdr_background_{}2.png'.format(noise_type))

    # sorted_sdrs = sorted(sdr_dict.items(), key=operator.itemgetter(1))[::-1]
    # f = open('../analysis_output/sdrs_sorted_bigger_dataset.txt', 'w')
    #
    # for sdr in sorted_sdrs:
    #     f.write(str(sdr[1]) + '\t' + str(sdr[0]) + '\n')
    # f.close()




if __name__ == '__main__':
    main()
