# Sound Separation Driver

# enter in command line before running this:
# export DYLD_LIBRARY_PATH=/System/Library/Frameworks/Python.framework/Versions/2.7/lib:$DYLD_LIBRARY_PATH

import numpy as np
import scipy
import math
import sklearn
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import Code.SourceSeparation as nussl
import os
from os.path import join, isfile, splitext
from time import time
import mir_eval
from sklearn.svm import SVR



def get_mixture_paths():
    folder = 'training/'
    return get_mixture_paths2(folder)


def get_mixture_paths2(folder):
    fileNames = {}
    ext = '.wav'
    for file in [f for f in os.listdir(folder) if isfile(join(folder, f))]:
        if splitext(file)[1] == ext:
            fgnd, bknd = file.split('__')
            fgnd += ext
            fileNames[join(folder, file)] = (fgnd, bknd)
    return fileNames


def get_foreground_paths():
    folder = 'foreground/'
    return [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))]


def get_background_paths():
    folder = 'background/'
    paths = []
    for bkgdFile in next(os.walk(folder))[2]:
        paths.append(join(folder, bkgdFile))
    return paths


def run_repet(path, do_sim):
    mixture = nussl.AudioSignal(path)
    repet_type = nussl.RepetType.ORIGINAL

    if do_sim:
        repet_type = nussl.RepetType.SIM

    repet = nussl.Repet(mixture, Type=repet_type)
    repet.Run()

    if do_sim:
        feature = repet.GetSimilarityMatrix()
    else:
        feature = repet.GetBeatSpectrum()

    return {'mixture': mixture, 'feature': feature, 'separation': repet.MakeAudioSignals()}


def compute_beat_spectrum_statistics(beat_spectrum):
    # normalize(beat_spectrum)
    return np.std(beat_spectrum), np.mean(beat_spectrum)


def training_beat():
    mixture_paths = get_mixture_paths()

    foreFolder = 'foreground/'
    max_dur = 441000

    r_stds = []
    r_mean = []
    sdrs = []
    for m_path, (f_path, b_path) in mixture_paths.items():
        r = run_repet(m_path, False)

        r_stats = compute_beat_spectrum_statistics(r['feature'])
        r_stds.append(r_stats[0])
        r_mean.append(r_stats[1])

        s = nussl.AudioSignal(join(foreFolder, f_path)).AudioData[0:max_dur] # original signal foreground
        se = r['separation'][0].AudioData[0:max_dur] # repet separated signal foreground

        mir_eval.separation.validate(s.T, se.T)
        sdrs.append(mir_eval.separation.bss_eval_sources(s.T, se.T)[0][0])

        # call bss_eval
        # global eng
        # sdrs.append(eng.bss_eval_sources(se, s)[0])

    return r_stds, r_mean, sdrs


def do_kNN_test(n_neighbors_max):
    mixture_test = 'mixture_test/'
    foreground_test = 'foreground/'

    mixes = get_mixture_paths2(mixture_test)
    max_dur = 441000

    print 'starting training'
    r_stds, r_mean, sdrs = training_beat()
    print 'finished training'

    r_stds = np.array(r_stds).reshape((len(r_stds), 1))
    sdrs = np.array(sdrs).reshape((len(sdrs), 1))

    # testing
    pred_std = []
    act = []
    for mix_path, (f_path, b_path) in mixes.items():
        r = run_repet(mix_path, False)
        cur_std = compute_beat_spectrum_statistics(r['feature'])[0]
        pred_std.append(cur_std)

        s = nussl.AudioSignal(join(foreground_test, f_path)).AudioData[0:max_dur]
        se = r['separation'][0].AudioData[0:max_dur]

        mir_eval.separation.validate(s.T, se.T)
        act.append(mir_eval.separation.bss_eval_sources(s.T, se.T)[0][0])


    NNcorr = []
    neighbs = range(1, n_neighbors_max + 1)
    for n in neighbs:
        knn = neighbors.KNeighborsRegressor(n)
        pred = []
        for std in pred_std:
            pred.append(knn.fit(r_stds, sdrs).predict(std)[0][0])
        NNcorr.append(scipy.stats.pearsonr(pred, act)[0])

    print NNcorr

    SVRcorr = []
    svr_types = ['rbf', 'linear']
    for s_type in svr_types:
        svr = SVR(kernel=s_type)
        pred = []
        for std in pred_std:
            pred.append(svr.fit(r_stds, sdrs).predict(std)[0])
        SVRcorr.append(scipy.stats.pearsonr(pred, act)[0])

    plt.plot(neighbs, NNcorr, 'ro')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('kNN Correlation')
    plt.xlim([0, n_neighbors_max+1])
    plt.ylim([min(NNcorr) - min(NNcorr)*0.2, max(NNcorr) + max(NNcorr)*0.2])
    ax = plt.axes()
    xa = ax.get_xaxis()
    xa.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig('kNN_corr1.png')
    plt.close()

    plt.plot([1,2], SVRcorr, 'ro')
    plt.xticks([0, 1], svr_types)
    plt.xlabel('Support Vector Regressor')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title('SVR Type')
    plt.margins(0.2)
    plt.ylim([min(SVRcorr) - min(SVRcorr)*0.2, max(SVRcorr) + max(SVRcorr)*0.2])
    ax = plt.axes()
    xa = ax.get_xaxis()
    xa.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig('SVR_corr.png')


def main():
    print 'starting kNN test'
    start = time()
    do_kNN_test(10)
    print time() - start, 'sec'


#######################################################################
if __name__ == "__main__":
    main()
