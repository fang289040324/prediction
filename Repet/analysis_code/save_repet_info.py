import numpy as np
import nussl
import os
from os.path import join, isfile, splitext
import pickle
import mir_eval
import matlab_wrapper
import bss_eval.bss_eval_wrapper as bss
from multiprocessing import Pool

MIR_EVAL = True
MATLAB_WRAPPER = False
MATLAB_ENGINE = False

SHOULD_PICKLE_BEAT_SPEC = True
SHOULD_PICKLE_SIM_MAT = False

max_sec = 10.0
sample_rate = 44100
max_dur = max_sec * sample_rate

base_folder = '../generated_audio/'
mixture_folder = base_folder + 'test/'
foreground_folder = base_folder + 'foreground/'
background_folder = base_folder + 'test_background/'
out = '../pickles_rolloff/'

def save_info():
    pool = Pool()
    pool.map(run_repet_and_pickle, get_mixture_paths2(mixture_folder).iteritems())
    pool.join()
    pool.close()


def run_repet_and_pickle(paths):
    mixture_path, (fgnd_name, bknd_name) = paths
    mixture_file = os.path.split(mixture_path)[1]
    mixture_pickle_name = splitext(mixture_file)[0]
    mixture_pickle_path = join(out, mixture_pickle_name)

    if not os.path.exists(mixture_pickle_path):
        os.mkdir(mixture_pickle_path)

    try:
        a = nussl.AudioSignal(mixture_path)
    except:
        print('Couldn\'t read {}'.format(file))
        return

    r = nussl.Repet(a)

    if SHOULD_PICKLE_BEAT_SPEC:
        beat_spec = r.get_beat_spectrum()
        pickle_file = join(mixture_pickle_path, '{}__beat_spec.pick'.format(mixture_pickle_name))
        pickle.dump(beat_spec, open(pickle_file, 'wb'))
        print 'pickled {} beat spectrum'.format(mixture_file)

    if SHOULD_PICKLE_SIM_MAT:
        sim_mat = r.get_similarity_matrix()
        pickle.dump(sim_mat, open(join(mixture_pickle_path, '{}__sim_mat.pick'.format(mixture_pickle_name)), 'wb'))
        print 'pickled {} sim matrix'.format(mixture_file)

    r()

    repet_bg, repet_fg = r.make_audio_signals()

    repet_bg.truncate_samples(max_dur)
    repet_fg.truncate_samples(max_dur)

    true_fg = nussl.AudioSignal(join(foreground_folder, fgnd_name))
    if true_fg.num_channels != 1:
        true_fg.audio_data = true_fg.get_channel(1)

    true_bg = nussl.AudioSignal(join(background_folder, bknd_name))
    if true_bg.num_channels != 1:
        true_bg.audio_data = true_bg.get_channel(1)

    true_fg.truncate_samples(max_dur)
    true_bg.truncate_samples(max_dur)

    estimated = np.array([repet_bg.get_channel(1), repet_fg.get_channel(1)])
    true_srcs = np.array([true_bg.get_channel(1), true_fg.get_channel(1)])

    mir_eval.separation.validate(true_srcs, estimated)
    bss_vals = mir_eval.separation.bss_eval_sources(true_srcs, estimated)


    sdr_dict = {'foreground': {'sdr': bss_vals[0][0], 'sir': bss_vals[1][0], 'sar': bss_vals[2][0]},
                'background': {'sdr': bss_vals[0][1], 'sir': bss_vals[1][1], 'sar': bss_vals[2][1]}}

    should_pickle_sdr = True
    if should_pickle_sdr:
        pickle.dump(sdr_dict, open(join(mixture_pickle_path, '{}__sdrs.pick'.format(mixture_pickle_name)), 'wb'))
        print 'pickled {} sdrs'.format(mixture_file)
    else:
        print '{} sdrs similar, not pickling'.format(mixture_file)

def get_foreground_paths(base_folder):
    folder = base_folder + 'foreground/'
    return [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))]


def get_background_paths(base_folder):
    folder = base_folder + 'background/'
    paths = []
    for bkgdFile in next(os.walk(folder))[2]:
        paths.append(join(folder, bkgdFile))
    return paths

def get_mixture_paths(base_folder):
    folder = base_folder + 'training/'
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


if __name__ == '__main__':
    save_info()
