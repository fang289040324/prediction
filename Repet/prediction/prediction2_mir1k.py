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
    pickle_folder = '../mir_1k/pickles'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if '__beat_spec.pick' in f]
    pickle_folders_to_load = sorted(pickle_folders_to_load)

    sdr_type = 'background'

    fits = []
    sdrs = []
    for pick in pickle_folders_to_load:
        pick = pick.replace('__beat_spec.pick', '')
        beat_spec_path = join(pickle_folder, pick + '__beat_spec.pick')
        beat_spec = pickle.load(open(beat_spec_path, 'rb'))

        entropy, log_mean = beat_spectrum_prediction_statistics(beat_spec)
        fit_X = [entropy, log_mean]
        fits.append(fit_X)

        sdrs_name = join(pickle_folder, pick + '__sdrs.pick')
        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        cur_sdr = sdr_vals[sdr_type][0]
        sdrs.append(cur_sdr)

    fits = np.array(fits)
    sdrs = np.array(sdrs).reshape(-1, 1)
    knn = neighbors.KNeighborsRegressor(5, weights='distance')
    scores = cross_validation.cross_val_predict(knn, fits, sdrs, cv=10, verbose=1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # knn.fit(fits, sdrs)


def beat_spectrum_prediction_statistics(beat_spectrum):
    beat_spec_norm = beat_spectrum / np.max(beat_spectrum)

    entropy = - sum(p * np.log(p) for p in np.abs(beat_spec_norm)) / len(beat_spec_norm)
    log_mean = np.log(np.mean(beat_spectrum[1:]))

    return entropy, log_mean



if __name__ == '__main__':
    main()