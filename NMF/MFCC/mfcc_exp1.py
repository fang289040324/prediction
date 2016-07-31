# coding=utf-8
"""
MFCC experiment 1
"""
from sklearn.cluster import KMeans
import librosa
import os
import numpy as np
from sklearn.decomposition import NMF
import mir_eval
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def main():
    """

    :return:
    """
    history = '../audio/flute1__band3_none_0.0.wav'
    sources = mfcc_clustering(history, 3)


def mfcc_clustering(file_name, n_clusters):
    """
    From Prem
    :return:
    """

    clusterer = KMeans(n_clusters=n_clusters)

    print(file_name)
    mix, sr = librosa.load(file_name)
    mix_stft = librosa.stft(mix)
    comps, acts = find_template(mix_stft, sr, 100, 101, 0, mix_stft.shape[1])
    cluster_comps = librosa.feature.mfcc(S=comps)[1:14]
    save_mfcc_img(file_name[:-4] + '_mfcc.png', np.flipud(cluster_comps))
    clusterer.fit_transform(cluster_comps.T)
    labels = clusterer.labels_
    # print(labels)
    sources = []

    for cluster_index in range(n_clusters):
        indices = np.where(labels == cluster_index)[0]
        template, residual = extract_template(comps[:, indices], mix_stft)
        t = librosa.istft(template)
        sources.append(t)

    return np.array(sources)


def save_mfcc_img(out_file_name, mfcc):
    """

    :param out_file_name:
    :return:
    """
    plt.close('all')
    i = plt.imshow(mfcc, cmap=cm.jet, interpolation='nearest', aspect='auto')
    plt.colorbar(i)
    # plt.grid(True)
    plt.gca().invert_yaxis()
    plt.title(out_file_name[:-9] + ' MFCC components')
    plt.ylabel('MFCC')
    plt.xlabel('NMF Dictionary Entry')
    plt.savefig(out_file_name)


def extract_template(comps, music_stft):
    """
    from Prem
    :param comps:
    :param music_stft:
    :return:
    """
    K = comps.shape[1]

    # initialize transformer (non-negative matrix factorization) with K components
    transformer = NMF(n_components=K, init='custom')

    # W and H are random at first
    W = np.random.rand(comps.shape[0], K)
    H = np.random.rand(K, music_stft.shape[1])

    # set W to be the template components you want to extract
    W[:, 0:K] = comps

    # don't let W get updated in the non-negative matrix factorization
    params = {'W': W, 'H': H, 'update_W': False}
    comps_music = transformer.fit_transform(np.abs(music_stft), **params)
    acts_music = transformer.components_

    # reconstruct the signal
    music_reconstruction = comps_music.dot(acts_music)

    # mask the input signal
    music_stft_max = np.maximum(music_reconstruction, np.abs(music_stft))
    mask = np.divide(music_reconstruction, music_stft_max)
    mask = np.nan_to_num(mask)

    # binary mask
    mask = np.round(mask)

    # template - extracted template, residual - everything that's leftover.
    template = np.multiply(music_stft, mask)
    residual = np.multiply(music_stft, 1 - mask)

    return template, residual


def find_template(music_stft, sr, min_t, n_components, start, end):
    """
    from Prem
    :param music_stft:
    :param sr:
    :param min_t:
    :param n_components:
    :param start:
    :param end:
    :return:
    """
    template_stft = music_stft[:, start:end]
    layer = librosa.istft(template_stft)
    layer_rms = np.sqrt(np.mean(layer * layer))

    comps = []
    acts = []
    errors = []

    for T in range(min_t, n_components):
        transformer = NMF(n_components=T)
        comps.append(transformer.fit_transform(np.abs(template_stft)))
        acts.append(transformer.components_)
        errors.append(transformer.reconstruction_err_)

    # knee = np.diff(errors, 2)
    # knee = knee.argmax() + 2
    knee = 0

    # print 'Using %d components' % (knee + min_t)
    return comps[knee], acts[knee]

if __name__ == "__main__":
    main()