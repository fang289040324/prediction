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
    noise_type_to_graph = stft_swap

    sdr_type = 'background'

    perfect_coef = 0.0
    first = True

    original_mixtures = [m for m in pickle_folders_to_load if float(m.split('_')[-1]) == perfect_coef]
    original_sdrs = []

    for mixture_folder in original_mixtures:
        sdrs_name = join(pickle_folder, mixture_folder, mixture_folder + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        original_sdrs.append(sdr_vals[sdr_type][0])

    original_names_to_graph = []
    hist = np.histogram(original_sdrs)
    for i in range(len(hist[0])):
        name = original_mixtures[ (np.abs(np.array(original_sdrs) - hist[1][i])).argmin()]
        name = name.split('_')[0] + '__' + name.split('_')[2]
        original_names_to_graph.append(name)

    original_sdr_tup = zip([0.0] * len(original_sdrs), original_sdrs)
    original_names_sdrs = {original_names_to_graph[i] : [original_sdr_tup[i]]
                           for i in range(len(original_names_to_graph))}

    for original_name in original_names_to_graph:
        perturbed_sdr_names = [p for p in pickle_folders_to_load
                               if p.find(original_name) != -1
                               and p.find(noise_type_to_graph) != -1]

        for perturbed_folder in perturbed_sdr_names:
            sdrs_name = join(pickle_folder, perturbed_folder, perturbed_folder + '__sdrs.pick')
            sdr_vals = pickle.load(open(sdrs_name, 'rb'))
            coef = float(perturbed_folder.split('_')[-1])
            original_names_sdrs[original_name].append((coef, sdr_vals[sdr_type][0]))


    plt.close('all')
    for name, coefs in original_names_sdrs.iteritems():
        coef, sdr = zip(*coefs)

        # sdr_values = zip(*[diffs for diffs in coefs.values()])
        # sdr_coefs = sorted(coefs_by_noise_type[noise_type])
        # sdr_coefs.insert(0, '')


        # plt.yscale('log')
        # plt.violinplot(sdr_values, showmeans=False, showmedians=True)
        plt.plot(coef, sdr)

        # plt.xticks(range(len(sdr_values)+1), sdr_coefs, rotation='vertical')
        plt.title('repetition perturbation coef vs. {sdr_type} SDR {noise_type} (individual)'.format(sdr_type=sdr_type,
                                                                                        noise_type=noise_type_to_graph))
        plt.xlabel('perturbation coefficient')
        plt.ylabel('SDR (dB)')
        # plt.subplots_adjust(bottom=0.15)
        plt.savefig(out + 'bigger_dataset_sdr_{sdr_type}_individual_{noise_type}.png'.format(sdr_type=sdr_type,
                                                                                       noise_type=noise_type_to_graph))





if __name__ == '__main__':
    main()
