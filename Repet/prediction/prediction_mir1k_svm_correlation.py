import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn import cross_validation
from sklearn.svm import SVR
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt
import nussl

def main():
    pickle_folder = '../mir_1k/pickles/'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if '__beat_spec.pick' in f]
    pickle_folders_to_load = sorted(pickle_folders_to_load)
    # pickle_folders_to_load = [p for p in pickle_folders_to_load if '-' not in p and '+' not in p]

    fg_or_bg = 'background'
    sdr_type = 'sdr'

    print 'num files = ', len(pickle_folders_to_load)

    n_folds = 10
    perm = np.random.permutation(len(pickle_folders_to_load))
    folds = np.array_split(perm, n_folds)

    true_sdrs = []
    pred_sdrs = []
    run = 1
    for fold_indices in folds:
        print 'Doing run ', run

        test_pickles = [pickle_folders_to_load[i] for i in fold_indices]
        train_pickles = [pickle_folders_to_load[i] for i in perm if i not in fold_indices]
        svr = neighbors.KNeighborsRegressor(5, weights='distance')

        fits = []
        sdrs = []
        for pick in train_pickles:
            pick = pick.replace('__beat_spec.pick', '')
            beat_spec_name = join(pickle_folder, pick + '__beat_spec.pick')
            beat_spec = pickle.load(open(beat_spec_name, 'rb'))

            fit_X = beat_spectrum_prediction_statistics(beat_spec)
            fits.append(fit_X)

            sdrs_name = join(pickle_folder, pick + '__sdrs.pick')
            sdr_vals = pickle.load(open(sdrs_name, 'rb'))
            cur_sdr = sdr_vals[fg_or_bg][sdr_type]
            sdrs.append(cur_sdr)

        fits = np.array(fits)
        sdrs = np.array(sdrs)
        svr.fit(fits, sdrs)
        print 'min = ', min(sdrs), ' max = ', max(sdrs)

        for pick in test_pickles:
            pick = pick.replace('__beat_spec.pick', '')
            beat_spec_name = join(pickle_folder, pick + '__beat_spec.pick')
            beat_spec = pickle.load(open(beat_spec_name, 'rb'))
            entropy, log_mean= beat_spectrum_prediction_statistics(beat_spec)

            fit_X = np.array([entropy, log_mean], ndmin=2)

            sdrs_name = join(pickle_folder, pick + '__sdrs.pick')
            sdr_vals = pickle.load(open(sdrs_name, 'rb'))
            cur_sdr = sdr_vals[fg_or_bg][sdr_type]

            guess = svr.predict(fit_X)

            true_sdrs.append(cur_sdr)
            pred_sdrs.append(guess)
        run += 1

    plt.plot(true_sdrs, pred_sdrs, '.', color='blue', label='data')
    plt.plot(np.linspace(-15.0, 15.0),np.linspace(-15.0, 15.0), '--', color='red', label='y=x')
    plt.plot(np.linspace(-15.0, 15.0),np.linspace(-10.0, 20.0), '--', color='green', label='+/- 5dB')
    plt.plot(np.linspace(-15.0, 15.0),np.linspace(-20.0, 10.0), '--', color='green')
    plt.title('MIR-1K Dataset')
    plt.xlabel('True SDR (dB)')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.ylabel('Predicted SDR (dB)')
    plt.legend(loc='lower right')
    plt.savefig('scatter_true_pred_mir1k_knn.png')


def beat_spectrum_prediction_statistics(beat_spectrum):
    beat_spec_norm = beat_spectrum / np.max(beat_spectrum)

    entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    log_mean = np.log(np.mean(beat_spectrum[1:]))

    third_stat = beat_spectrum_magic(beat_spectrum)

    # beat_spectrum = beat_spectrum[:1]
    # beat_spec_norm = beat_spectrum / np.max(beat_spectrum)
    #
    # entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    # log_mean = np.log(np.mean(beat_spec_norm))

    return [entropy, log_mean]

def beat_spectrum_magic(beat_spectrum):
    rep_period = nussl.Repet.find_repeating_period_complex(beat_spectrum)

    n_periods = float(len(beat_spectrum)) / float(rep_period)

    n_steps = int(n_periods)

    diff = 0
    for step in range(2, n_steps):
        index = step * rep_period
        low = index - (rep_period / 2)
        high = index + (rep_period / 2)
        diff += np.abs(np.argmax(beat_spectrum[low:high]) - index)

    diff = float(diff) / n_periods

    # strength = sum(beat_spectrum[step] / beat_spectrum[rep_period] for step in range(n_steps)) / n_periods

    return diff


if __name__ == '__main__':
    main()