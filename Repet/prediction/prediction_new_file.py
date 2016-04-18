import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from os.path import join
import matplotlib.pyplot as plt
import nussl
import mir_eval

def main():
    pickle_folder = '../pickles'
    pickle_folders_to_load = [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]
    pickle_folders_to_load = sorted(pickle_folders_to_load)
    pickle_folders_to_load = [p for p in pickle_folders_to_load if 'drums1__' not in p]

    test_folder = '../test_audio'
    test_mixes_folder = test_folder + '/mixtures/'
    test_bkgnd_folder = test_folder + '/backgrounds/'
    test_files_to_load = [f for f in os.listdir(test_folder) if os.path.isfile(join(test_folder, f))
                          if os.path.splitext(join(test_folder, f)) == 'wav']

    num_runs = 5
    sdr_type = 'background'
    neighbor_mult = 2
    done = False

    for run in range(num_runs):
        n_neighbors = (run + 1) * neighbor_mult
        knn = neighbors.KNeighborsRegressor(n_neighbors)
        print n_neighbors, 'neighbors:'

        fits = []
        sdrs = []
        for pick in pickle_folders_to_load:
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

        # for test_file in test_files_to_load:
        test_mixture_file = test_mixes_folder + 'FooFighters-AllMyLife_mix_excerpt.wav'
        test_bkgnd_file = test_bkgnd_folder + 'FooFighters-AllMyLife_background.wav'

        back_signal = nussl.AudioSignal(test_bkgnd_file)

        mixture_signal = nussl.AudioSignal(test_mixture_file)
        repet = nussl.Repet(mixture_signal)
        repet.plot(test_folder + '/repet_output/all_my_life_beat_spec.png')
        beat_spectrum = repet.get_beat_spectrum()
        entropy, log_mean = beat_spectrum_prediction_statistics(beat_spectrum)
        fit_X = np.array([entropy, log_mean], ndmin=2)
        est_sdr = knn.predict(fit_X)[0][0]

        repet()

        est_bknd, est_fgnd = repet.make_audio_signals()
        est_bknd.write_audio_to_file(test_folder + '/repet_output/all_my_life_back.wav')
        est_fgnd.write_audio_to_file(test_folder + '/repet_output/all_my_life_fore.wav')

        mir_eval.separation.validate(back_signal.audio_data, est_bknd.audio_data)
        true_sdr = zip(*mir_eval.separation.bss_eval_sources(back_signal.audio_data, est_bknd.audio_data))[0]

        print 'Estimated SDR= ', est_sdr,  ' ::: True SDR=', true_sdr[0]
        return





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