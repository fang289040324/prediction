import nussl
import mir_eval
import numpy as np
import os
from os.path import isfile, splitext, join
from multiprocessing import Pool
import pickle
import copy


def main():
    """

    :return:
    """
    mir_1k_folder = 'MIR-1K/UndividedWavfile/'
    mir_file_paths = get_wav_paths_in_folder(mir_1k_folder)
    np.random.seed(0)
    repet_db_range = 5.0
    duet_range = 0.8
    n_times = 10
    paths_and_dbs = [(mir_file_paths[i],
                      (np.random.random() - 0.5) * 2 * repet_db_range,
                      (np.random.random() - 0.5) * duet_range,
                      label)
                     for i in range(len(mir_file_paths))
                     for label in range(n_times)]

    # uncomment this to debug
    # for file, db1, db2, label in paths_and_dbs:
    #     run_repet_duet_and_pickle(file, db1, db2, label)

    # comment this to debug
    pool = Pool()
    pool.map(run_wrapper, paths_and_dbs)
    pool.close()


def run_wrapper(args):
    """

    :param args:
    :return:
    """
    return run_repet_duet_and_pickle(*args)


def run_repet_duet_and_pickle(file_path, repet_db_change, duet_change, label):
    """

    :param file_path:
    :param repet_db_change:
    :param duet_change: change in attenuation between
    :param label: label is for uniqueness of file names with multi-threading
    :return:
    """
    pickle_output_folder = 'pickles_combined/'
    audio_output_folder = 'audio_combined/'
    file_name = os.path.split(file_path)[1]
    pickle_name = splitext(file_name)[0]

    signal = nussl.AudioSignal(file_path)
    music = signal.get_channel(1).reshape(-1, 1).T # channel 1 is background (music)
    singing = signal.get_channel(2).reshape(-1, 1).T # channel 2 is foreground (singing)

    signal.audio_data = do_manipulation(music, singing, repet_db_change, duet_change)
    signal.write_audio_to_file(join(audio_output_folder, '{0}_{1}.wav'.format(file_name, label)))

    beat_spec, repet_sdr_dict = get_repet_beat_spec_and_sdrs(signal)
    duet_hist, duet_sdr_dict = get_duet_histogram_and_sdrs(signal)

    pickle_dict = {'file_name': signal.file_name, 'label': label,
                   'repet_change': repet_db_change, 'duet_change': duet_change,
                   'beat_spec': beat_spec, 'repet_sdr_dict': repet_sdr_dict,
                   'duet_hist': duet_hist, 'duet_sdr_dict': duet_sdr_dict}

    pickle.dump(pickle_dict, open(join(pickle_output_folder, '{0}_{1}.pick'.format(pickle_name, label)), 'wb'))
    print('pickled {0} {1}'.format(pickle_name, label))


def get_repet_beat_spec_and_sdrs(audio_signal):
    """

    :param audio_signal:
    :return:
    """
    repet = nussl.Repet(audio_signal, do_mono=True)
    beat_spec = repet.get_beat_spectrum()

    repet()
    est_bg, est_fg = repet.make_audio_signals()

    estimated = np.array([est_bg.get_channel(1), est_fg.get_channel(1)])
    true_srcs = np.array([audio_signal.get_channel(1), audio_signal.get_channel(2)])

    return beat_spec, run_bss_eval(true_srcs, estimated)


def get_duet_histogram_and_sdrs(audio_signal):
    """

    :param audio_signal:
    :return:
    """
    duet = nussl.Duet(audio_signal, num_sources=2)
    duet()

    duet_hist = duet.non_normalized_hist
    src1, src2 = duet.make_audio_signals()

    estimated = np.array([src1.get_channel(1), src2.get_channel(1)])
    true_srcs = np.array([audio_signal.get_channel(1), audio_signal.get_channel(2)])

    return duet_hist, run_bss_eval(true_srcs, estimated)


def do_manipulation(music, singing, repet_change, duet_change):
    """

    :param music:
    :param singing:
    :param repet_change:
    :param duet_change:
    :return:
    """

    repet_change_singing = 10.0**(repet_change / 20.0) * singing
    attenuation_ch1 = float(duet_change + 1) / 2
    attenuation_ch2 = 1 - attenuation_ch1

    singing_ch1 = attenuation_ch1 * repet_change_singing
    singing_ch2 = attenuation_ch2 * repet_change_singing

    music_ch1 = attenuation_ch2 * music
    music_ch2 = attenuation_ch1 * music

    ch1 = singing_ch1 + music_ch1
    ch2 = singing_ch2 + music_ch2
    return np.vstack((ch1, ch2))


def run_bss_eval(true, estimated):
    """

    :param estimated:
    :param true:
    :return:
    """

    index = np.where(estimated.max(1) == 0)
    if len(index[0]) > 0:
        # Can't pass a silent source to mir_eval
        print("silent source")
        estimated[index[0][0]][0] += 1

    mir_eval.separation.validate(true, estimated)


    bss_vals = mir_eval.separation.bss_eval_sources(true, estimated)

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


def get_pickle_object(path_to_pickle):
    """

    :param path_to_pickle:
    :return:
    """
    pick = pickle.load(open(path_to_pickle, 'rb'))

if __name__ == '__main__':
    main()