# Sound Separation Driver

# enter in command line before running this:
# export DYLD_LIBRARY_PATH=/System/Library/Frameworks/Python.framework/Versions/2.7/lib:$DYLD_LIBRARY_PATH

import sys
import numpy as np
import scipy
import math
# import matlab.engine
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import Code.SourceSeparation as nussl
import os
from os.path import join, isfile, splitext
import subprocess

'''
4 Major components:
1) REPET - get either beat spectrum or similarity matrix
2) BSS_EVAL to use beat spectrum or similarity matrix to get SDR
3) Using different classifiers, train on this SDR data
4) Compare output from different classifiers

Matlab functions in BSS_EVAL:
####################### Function 1 #######################
[SDR,SIR,SAR,perm]=bss_eval_sources(se,s)
% Inputs:
% se: nsrc x nsampl matrix containing estimated sources
% s: nsrc x nsampl matrix containing true sources
%
% Outputs:
% SDR: nsrc x 1 vector of Signal to Distortion Ratios
% SIR: nsrc x 1 vector of Source to Interference Ratios
% SAR: nsrc x 1 vector of Sources to Artifacts Ratios
% perm: nsrc x 1 vector containing the best ordering of estimated sources
% in the mean SIR sense (estimated source number perm(j) corresponds to
% true source number j)

####################### Function 2 #######################
% [MER,perm]=bss_eval_mix(Ae,A)
% Inputs:
% Ae: either a nchan x nsrc estimated mixing matrix (for instantaneous
% mixtures) or a nchan x nsrc x nbin estimated frequency-dependent mixing
% matrix (for convolutive mixtures)
% A: the true nchan x nsrc or nchan x nsrc x nbin mixing matrix
%
% Outputs:
% MER: nsrc x 1 vector of Mixing Error Ratios (SNR-like criterion averaged
% over frequency and expressed in decibels, allowing arbitrary scaling for
% each source in each frequency bin)
% perm: nsrc x 1 vector containing the best ordering of estimated sources
% in the maximum MER sense (estimated source number perm(j) corresponds to
% true source number j)

####################### Function 3 #######################
[SDR,ISR,SIR,SAR,perm]=bss_eval_images(ie,i)
% Inputs:
% ie: nsrc x nsampl x nchan matrix containing estimated source images
% i: nsrc x nsampl x nchan matrix containing true source images
%
% Outputs:
% SDR: nsrc x 1 vector of Signal to Distortion Ratios
% ISR: nsrc x 1 vector of source Image to Spatial distortion Ratios
% SIR: nsrc x 1 vector of Source to Interference Ratios
% SAR: nsrc x 1 vector of Sources to Artifacts Ratios
% perm: nsrc x 1 vector containing the best ordering of estimated source
% images in the mean SIR sense (estimated source image number perm(j)
% corresponds to true source image number j)
'''


def func(A, B):  # comparematrices(A,B):
    colA = A.shape[1]
    colB = B.shape[1]

    # method 1 - n is small dim, m is larger, matnew is new comparison matrix
    if colA == colB and colA != 1:
        Aprime = normalize(A, axis=1, norm='l2')
        Bprime = normalize(B, axis=1, norm='l2')
        if colA == 1:
            dist = np.linalg.norm(Aprime - Bprime)  # L2 norm (vectors)
        else:
            dist = np.linalg.norm(Aprime - Bprime, 2)  # Frobenius norm (matrices)
    else:
        if colA < colB:
            n = colA
            m = colB
            big = B
            small = A
        else:
            n = colB
            m = colA
            big = A
            small = B
        matnew = np.identity(m)
        matnew[0:n, 0:n] = small
        bigprime = normalize(big, axis=1, norm='l2')
        matnewprime = normalize(matnew, axis=1, norm='l2')
    dist = np.linalg.norm(matnewprime - bigprime, 2)

    print dist


def get_mixture_paths():
    folder = 'training/'
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
    #normalize(beat_spectrum)
    return np.std(beat_spectrum), np.mean(beat_spectrum)


def training(eng, do_sim):
    mixture_paths = get_mixture_paths()
    foreground_paths = get_foreground_paths()
    background_paths = get_background_paths()


    r_stds = []
    r_mean = []
    for m_path, (f_path, b_path) in mixture_paths.items():
        r = run_repet(m_path, do_sim)

        if do_sim:
            r_stats = compute_beat_spectrum_statistics(r['feature'])
            r_stds.append(r_stats[0])
            r_mean.append(r_stats[1])



        # call bss_eval
        # SDR, SIR, SAR, perm = eng.bss_eval_sources(se, s)


    plt.plot(r_mean, r_stds, 'ro')
    plt.xlabel('Beat spectrum mean')
    plt.ylabel('Beat spectrum std deviation')
    plt.savefig('beat_spec.png')

    return 0


def main():
    do_sim = True
    #eng = matlab.engine.start_matlab()
    eng = 1
    training(eng, do_sim)





#######################################################################
if __name__ == "__main__":
    # cmd = 'export DYLD_LIBRARY_PATH=/System/Library/Frameworks/Python.framework/Versions/2.7/lib:$DYLD_LIBRARY_PATH'
    # subprocess.call(cmd, shell=True)

    main()
# # for each classifier:
#     data = sys.argv[1]  # what form is this input?
#     eng = matlab.engine.start_matlab()  # initiate matlab engine
#
#
#     # split into test/train data: example1 = [mixture, foreground, background]
#     train =
#     test =
#
#     beatmats = []
#     simmats = []
#     est_sources_mat = []
#     # run REPET on training set,
#     for i in range(len(train) + len(test)):
#         beatmats.append([])  # fill in matrix for each example in test and train sets
#         simmats.append([])  # each element of beat/simmats will be a matrix for example i
#         est_sources_mat.append([])  # fill in A' for each example
#     # get beat spectrum, similarity matrix
#     train_beatmats = beatmats[0:len(train)]
#     test_beatmats = beatmats[-len(test):-1]  # the rest of the examples
#     train_simmats = simmats[0:len(train)]
#     test_simmats = simmats[-len(test):-1]
#
#     se = est_sources  # estimated source (A')
#     s = train[column1]  # true source (A)
#     ie =  # estimated source images (A'image)
#     i =  # true source image (A image)
#     SDRmat = []
#     SDR, SIR, SAR, perm = eng.bss_eval_sources(se, s)
#     SDRi, ISR, SIRi, SARi, permi = eng.bss_eval_images(ie, i)
#     SDRmat.append([SDRi, SDR])  # SDR-beat spectrum, SDR-sim matrix
#     SDRmat = np.asarray(SDRmat)
#     '''
#     # plug in beat spectrum and similarity matrix to BSS EVAL to get SDRs - using Matlab Engine
#     # se = estimated sources (nsrc x n samples matrix)
#     # s = true sources (nsrc x n samples matrix - true)
#     # ie = nsrc x nsampl x nchan matrix containing estimated source images
#     # i = nsrc x nsampl x nchan matrix containing true source images
#     '''
#     SDR, SIR, SAR, perm = eng.bss_eval_sources(se, s)  # <- beat spectrum input
#     SDRi, ISR, SIRi, SARi, permi = eng.bss_eval_images(ie, i)  # <- similarity matrix input
#
#     ###### METHODS ###### Compare matrices
#     ## K Nearest Neighbors: define distance metric "func", "pyfunc" calls it I think?
#     neigh_beat = sklearn.neighbors.KNeighborsRegressor(k, weights='uniform', algorithm='auto', metric='pyfunc')
#     neigh_beat.fit(train_beatmats, SDR[:, 0])
#     SDRe_kNN_beat = neigh_beat.predict(test_beatmats[:, 0])
#     neigh_sim = sklearn.neighbors.KNeighborsRegressor(k, weights='uniform', algorithm='auto', metric='pyfunc')
#     neigh_sim.fit(train_simmats, SDR[:, 1])
#     SDRe_kNN_sim = neigh_sim.predict(test_simmats)
#
#     corr_mat = []
# corr1tmp = scipy.stats.pearsonr(SDRe_kNN_beat, SDRmat[:, 0])
# corr2tmp = scipy.stats.pearsonr(SDRe_kNN_sim, SDRmat[:, 1])
# corr_mat.append(corr1tmp, corr2tmp)
#
# ## SVM
# clf = svm.SVR()  # doesn't seem to recognize svm in sklearn
# clf.fit(train_beatmats, SDRmat[:, 0])
# SDRe_SVM_beat = clf.predict(test_beatmats)
# # clf2 = svm.SVR()
# # clf2.fit(train_simmats, SDRmat[:,1])  #redundant?
# clf.fit(train_simmats, SDRmat[:, 1])
# SDRe_SVM_sim = clf.predict(test_simmats)
#
# corr1tmp = scipy.stats.pearsonr(SDRe_SVM_beat, SDRmat[:, 0])
# corr2tmp = scipy.stats.pearsonr(SDRe_SVM_sim, SDRmat[:, 1])
# corr_mat.append(corr1tmp, corr2tmp)
#
# ## Bernoulli Restricted Boltzmann Machine
# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# RBM = BernoulliRBM(n_components=2)  # not sure how many
# RBM.fit(test_beatmats)
# SDRe_NN_beat = RBM.predict(test_beatmats)
# RBM.fit(test_simmats)
# SDRe_NN_sim = RBM.predict(test_simmats)
#
# corr1tmp = scipy.stats.pearsonr(SDRe_NN_beat, SDRmat[:, 0])
# corr2tmp = scipy.stats.pearsonr(SDRe_NN_sim, SDRmat[:, 1])
# corr_mat.append(corr1tmp, corr2tmp)
#
# x = np.ones(3)
# plt.figure(1)
# plt.scatter(x, corr_mat[:, 0])
# plt.scatter(2 * x, corr_mat[:, 1])  # how do you make this just two columns?
# plt.title('Correlation Coefficients for Each ML Algorithm/Repetition Measure Combination')
# plt.xlabel('Beat Spectrum, Similarity Matrix')
# plt.ylabel('Correlation Coeffient between real/predicted values')
#
# ## look at which gives the best correlation and plot that one
# plt.figure(2)
# plt.scatter(SDRmat[:, 0], SDRe_NN_beat)  # Neural Net, beat spectrum, for example
# plt.title('Best Combination')
# plt.ylabel('Predicted SDR by NN')
# plt.xlabel('True SDR by REPET')
# plt.show()
