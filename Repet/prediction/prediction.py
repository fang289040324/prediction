import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn import cross_validation
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt

def main():
    pickle_folder = '../pickles'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)
    pickle_folders_to_load = [p for p in pickle_folders_to_load if 'drums1__' not in p]

    percent_test = 0.15
    n_test_examples = int(len(pickle_folders_to_load) * percent_test)
    print str(len(pickle_folders_to_load) - n_test_examples), 'training, ', str(n_test_examples), 'test'
    num_runs = 5
    sdr_type = 'background'

    all_diffs = []
    neighbor_mult = 2

    for run in range(num_runs):
        print 'Doing run ', run + 1
        perm = np.random.permutation(len(pickle_folders_to_load))
        test_pickles = [pickle_folders_to_load[i] for i in perm[:int(len(perm) * percent_test)]]
        train_pickles = [pickle_folders_to_load[i] for i in perm[int(len(perm) * percent_test):]]

        n_neighbors = (run + 1) * neighbor_mult
        knn = neighbors.KNeighborsRegressor(5, weights='distance')

        fits = []
        sdrs = []
        for pick in train_pickles:
            beat_spec_name = join(pickle_folder, pick, pick + '__beat_spec.pick')
            beat_spec = pickle.load(open(beat_spec_name, 'rb'))

            entropy, log_mean = beat_spectrum_prediction_statistics(beat_spec)
            fit_X = [entropy, log_mean]
            fits.append(fit_X)

            sdrs_name = join(pickle_folder, pick, pick + '__sdrs.pick')
            sdr_vals = pickle.load(open(sdrs_name, 'rb'))
            cur_sdr = sdr_vals[sdr_type][0]
            sdrs.append(cur_sdr)

        fits = np.array(fits)
        sdrs = np.array(sdrs).reshape(-1, 1)
        knn.fit(fits, sdrs)

        diffs = []
        scores = []
        for pick in test_pickles:
            beat_spec_name = join(pickle_folder, pick, pick + '__beat_spec.pick')
            beat_spec = pickle.load(open(beat_spec_name, 'rb'))
            entropy, log_mean = beat_spectrum_prediction_statistics(beat_spec)

            fit_X = np.array([entropy, log_mean], ndmin=2)

            sdrs_name = join(pickle_folder, pick, pick + '__sdrs.pick')
            sdr_vals = pickle.load(open(sdrs_name, 'rb'))
            cur_sdr = sdr_vals[sdr_type][0]

            guess = knn.predict(fit_X)[0][0]

            diffs.append(cur_sdr - guess)

        all_diffs.append(np.array(diffs))

    # plt.boxplot(all_diffs, vert=True)
    plt.violinplot(all_diffs)
    plt.grid(axis='y')
    plt.title('Difference between KNN predicted SDR and true SDR - No drums, weight=\'distance\'')
    plt.ylabel('True SDR - Predicted SDR (dB)')
    plt.xlabel('Run #')
    plt.xticks(range(1, 6), [str(i) for i in range(1, 6)])
    plt.savefig('violin_plot_n_neighbors_no_drum_weight_distance.png')

    i = 1
    for diff_list in all_diffs:
        std = np.std(diff_list)
        print 'Number neighbors = ', str(i*neighbor_mult)
        print 'Mean = {0:.2f} dB'.format(np.mean(diff_list)), ' Std. Dev. = {0:.2f} dB'.format(std),
        print ' Min = {0:.2f} dB'.format(np.min(diff_list)), ' Max = {0:.2f} dB'.format(np.max(diff_list)),
        print ' ==== % more than 2 std = {0:.2f}%'.format(float(sum([1 for n in diff_list if np.abs(n) >= 2 * std]))
                                                     / float(len(diff_list)) * 100)
        i += 1

def beat_spectrum_prediction_statistics(beat_spectrum):
    beat_spec_norm = beat_spectrum / np.max(beat_spectrum)

    entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    log_mean = np.log(np.mean(beat_spectrum[1:]))
    # beat_spectrum = beat_spectrum[:1]
    # beat_spec_norm = beat_spectrum / np.max(beat_spectrum)
    #
    # entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    # log_mean = np.log(np.mean(beat_spectrum))

    return entropy, log_mean



if __name__ == '__main__':
    main()