# coding=utf-8
"""
Generate NMF MFCCs for the Generated REPET data.
"""
import nussl
import mir_eval
import numpy as np
import os
from os.path import isfile, splitext, join
from multiprocessing import Pool
import pickle
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import librosa

base_folder = '../../Repet/generated_audio/'
background_folder = base_folder + 'test_background/'
foreground_folder = base_folder + 'foreground/'
output_folder = '../mfcc_pickles/'
n_clusters = 2


def main():
    """

    :return:
    """

    mix_input_paths = get_mixture_paths2(base_folder + 'test/')


    # for key, value in mix_input_paths.iteritems():
    #     run_nmf_mfcc_and_pickle_mir_evals((key, value),
    #                                       n_clusters,
    #                                       output_folder)

    pool = Pool()
    pool.map(run_nmf_mfcc_and_pickle_mir_evals, mix_input_paths.iteritems())
    pool.close()

def run_wrapper(args):
    """

    :param args:
    :return:
    """
    return run_nmf_mfcc_and_pickle_mir_evals(*args)


def run_nmf_mfcc_and_pickle_mir_evals(paths):
    """

    :return:
    """
    mix_path, (fore_name, back_name) = paths
    file_name = os.path.split(mix_path)[1]
    pickle_name = splitext(file_name)[0]
    pickle_output_path = os.path.join(output_folder, pickle_name)

    if os.path.exists(pickle_output_path):
        print(pickle_output_path + 'exists! Skipping...')
        return

    back_path = os.path.join(background_folder, back_name)
    true_back = librosa.load(back_path)[0]
    fore_path = os.path.join(foreground_folder, fore_name)
    true_fore = librosa.load(fore_path)[0]

    estimated_sources, mfcc_clusters = do_nmf_and_clustering(mix_path, n_clusters)
    # transpose?
    min_len = np.min((len(true_fore), len(true_back), estimated_sources.shape[1]))
    true_back = true_back[:min_len]
    true_fore = true_fore[:min_len]
    true_srcs = np.vstack([true_back, true_fore])
    estimated_sources = estimated_sources[:, :min_len]
    sdr_dict = run_bss_eval(true_srcs, estimated_sources)

    pickle_dict = {'file_name': file_name, 'mfcc_clusters': mfcc_clusters, 'sdr_dict': sdr_dict}
    pickle.dump(pickle_dict, open(os.path.join(output_folder, pickle_output_path + '.pick'), 'wb'))
    print('pickled {}'.format(pickle_name))


def do_nmf_and_clustering(input_file, n_clusters):
    """

    :return:
    """
    clusterer = KMeans(n_clusters=n_clusters)

    mix, sr = librosa.load(input_file)
    mix_stft = librosa.stft(mix)
    comps, acts = find_template(mix_stft, sr, 100, 101, 0, mix_stft.shape[1])
    cluster_comps = librosa.feature.mfcc(S=comps)[1:14]
    clusterer.fit_transform(cluster_comps.T)
    labels = clusterer.labels_
    sources = []

    for cluster_index in range(n_clusters):
        indices = np.where(labels == cluster_index)[0]
        template, residual = extract_template(comps[:, indices], mix_stft)
        t = librosa.istft(template)
        sources.append(t)

    return np.array(sources), cluster_comps


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


def run_bss_eval(true_, estimated):
    """

    :param estimated:
    :param true_:
    :return:
    """

    index = np.where(estimated.max(1) == 0)
    if len(index[0]) > 0:
        # Can't pass a silent source to mir_eval
        print("silent source")
        estimated[index[0][0]][0] += 1

    mir_eval.separation.validate(true_, estimated)

    bss_vals = mir_eval.separation.bss_eval_sources(true_, estimated)

    sdr_dict = {'foreground': {'sdr': bss_vals[0][0], 'sir': bss_vals[1][0], 'sar': bss_vals[2][0]},
                'background': {'sdr': bss_vals[0][1], 'sir': bss_vals[1][1], 'sar': bss_vals[2][1]}}

    return sdr_dict


def get_wav_paths_in_folder(folder):
    """

    :param folder:
    :return:
    """
    return [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))
            and splitext(join(folder, f))[1] == '.wav']



def get_mixture_paths(base_folder):
    """

    :param base_folder:
    :return:
    """
    folder = base_folder + 'training/'
    return get_mixture_paths2(folder)


def get_mixture_paths2(folder):
    """

    :param folder:
    :return:
    """
    fileNames = {}
    ext = '.wav'
    for file in [f for f in os.listdir(folder) if isfile(join(folder, f))]:
        if splitext(file)[1] == ext:
            fgnd, bknd = file.split('__')
            fgnd += ext
            fileNames[join(folder, file)] = (fgnd, bknd)
    return fileNames

if __name__ == '__main__':
    main()