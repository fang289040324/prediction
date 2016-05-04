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
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    print str(len(pickle_folders_to_load)), 'total files'
    fg_or_bg = 'background'
    sdr_type = 'sdr'

    all_diffs = []

    n_folds = 10
    perm = np.random.permutation(len(pickle_folders_to_load)) # random permutation of indices
    folds = np.array_split(perm, n_folds) # splits into folds

    run = 1
    for fold_indices in folds:
        print 'Doing run ', run

        # convert indices to file names
        test_pickles = [pickle_folders_to_load[i] for i in fold_indices]
        train_pickles = [pickle_folders_to_load[i] for i in perm if i not in fold_indices]

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
            cur_sdr = sdr_vals[fg_or_bg][sdr_type]
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
            cur_sdr = sdr_vals[fg_or_bg][sdr_type]

            guess = knn.predict(fit_X)[0][0]

            diffs.append(cur_sdr - guess)

        all_diffs.append(np.array(diffs))
        run += 1

    # these two lines make it look pretty
    plt.style.use('bmh')
    plt.hist(all_diffs, histtype='stepfilled', stacked=True, alpha=0.8, bins=30)

    plt.title('Generated data histogram')
    plt.xlabel('True SDR $-$ Predicted SDR (dB)')
    plt.savefig('histogram_gen_bss_cross10_back_both_have_1st_val.png')

    # print out statistics about each of the runs
    mean, std1, std2 = [], [], []
    i = 1
    for diff_list in all_diffs:
        std = np.std(diff_list)
        mean.append(np.mean(diff_list))
        std1.append(std)
        per = float(sum([1 for n in diff_list if np.abs(n) >= 2 * std]))  / float(len(diff_list)) * 100
        std2.append(per)
        print 'Run ', str(i)
        print 'Mean = {0:.2f} dB'.format(np.mean(diff_list)), ' Std. Dev. = {0:.2f} dB'.format(std),
        print ' Min = {0:.2f} dB'.format(np.min(diff_list)), ' Max = {0:.2f} dB'.format(np.max(diff_list)),
        print ' ==== % more than 2 std = {0:.2f}%'.format(per)
        i += 1

    print '=' * 80
    print 'Avg. Mean = {0:.2f} dB'.format(np.mean(mean)), 'Avg. Std. Dev = {0:.2f} dB'.format(np.mean(std1)),
    print 'Avg. % more than 2 std = {0:.2f}%'.format(np.mean(std2))

    print 'max =', np.max(np.array(all_diffs)), ' min =', np.min(np.array(all_diffs))


def beat_spectrum_prediction_statistics(beat_spectrum):
    beat_spec_norm = beat_spectrum / np.max(beat_spectrum)
    # beat_spec_norm = beat_spec_norm[1:]
    entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    # log_mean = np.log(np.mean(beat_spectrum[1:] / np.max(beat_spectrum[1:])))
    log_mean = np.log(np.mean(beat_spectrum / np.max(beat_spectrum)))

    return entropy, log_mean



if __name__ == '__main__':
    main()