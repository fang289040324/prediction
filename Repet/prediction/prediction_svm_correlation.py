import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR
from sklearn import cross_validation
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def main():
    pickle_folder = '../pickles_rolloff'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)
    # pickle_folders_to_load = [p for p in pickle_folders_to_load if 'drums1__' not in p]

    print str(len(pickle_folders_to_load)), 'total'
    num_runs = 5
    fg_or_bg = 'background'
    sdr_type = 'sdr'

    all_diffs = []

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
        print 'min = ', min(sdrs), 'max = ', max(sdrs)

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

            true_sdrs.append(cur_sdr)
            pred_sdrs.append(guess)

        all_diffs.append((np.array(diffs)**2).mean())
        run += 1

    # plt.hexbin(true_sdrs, pred_sdrs, gridsize=40, cmap=plt.cm.Blues)
    plt.plot(true_sdrs, pred_sdrs, '.', color='blue', label='data')
    plt.plot(np.linspace(-50.0, 50.0),np.linspace(-50.0, 50.0), '--', color='red', label='y=x')
    plt.plot(np.linspace(-50.0, 50.0),np.linspace(-45.0, 55.0), '--', color='green', label='+/- 5dB')
    plt.plot(np.linspace(-50.0, 50.0),np.linspace(-55.0, 45.0), '--', color='green')
    plt.title('Generated data')
    plt.xlabel('True SDR (dB)')
    plt.xlim(-15, 40)
    plt.ylim(-15, 40)
    plt.ylabel('Predicted SDR (dB)')
    plt.legend(loc='lower right')
    # plt.colorbar()
    plt.savefig('scatter_true_pred_gen_knn.png')


def beat_spectrum_prediction_statistics(beat_spectrum):
    beat_spec_norm = beat_spectrum / np.max(beat_spectrum)

    entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    # log_mean = np.log(np.mean(beat_spectrum[1:]))
    log_mean = np.log(np.mean(beat_spectrum[1:] / np.max(beat_spectrum[1:])))


    # beat_spectrum = beat_spectrum[:1]
    # beat_spec_norm = beat_spectrum / np.max(beat_spectrum)
    #
    # entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    # log_mean = np.log(np.mean(beat_spectrum))

    return entropy, log_mean



if __name__ == '__main__':
    main()