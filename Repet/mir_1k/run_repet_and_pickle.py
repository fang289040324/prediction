import nussl
import mir_eval
import numpy as np
import os
from os.path import isfile, splitext, join
from multiprocessing import Pool
import pickle



def main():
    mir_1k_folder = 'MIR-1K/UndividedWavfile/'
    mir_file_paths = get_wav_paths_in_folder(mir_1k_folder)
    db_range = 5.0
    n_times = 10
    paths_and_dbs = [(mir_file_paths[i], (np.random.random() - 1) * db_range)
                     for i in range(len(mir_file_paths))
                     for j in range(n_times)]

    # for file, db in paths_and_dbs:
    #     run_repet_and_pickle(file, db)

    pool = Pool()
    pool.map(run_repet_wrapper, paths_and_dbs)
    pool.join()
    pool.close()

def run_repet_wrapper(args):
    return run_repet_and_pickle(*args)

def run_repet_and_pickle(file_path, db_change):
    output_folder = 'pickles_random/'
    file_name = os.path.split(file_path)[1]
    pickle_name = splitext(file_name)[0]

    sig = nussl.AudioSignal(file_path)
    true_bg = sig.get_channel(1).reshape(-1, 1).T # channel 1 is background (music)
    true_fg = sig.get_channel(2).reshape(-1, 1).T # channel 2 is foreground (singing)

    sig.audio_data = true_bg + 10.0**(db_change / 20.0) * true_fg

    repet = nussl.Repet(sig)

    num = '%.2f' % db_change
    num = '+' + num if db_change > 0.0 else num
    pickle_name += '_' + num

    beat_spec = repet.get_beat_spectrum()
    pickle_file = join(output_folder, '{}__beat_spec.pick'.format(pickle_name))
    pickle.dump(beat_spec, open(pickle_file, 'wb'))
    print 'pickled {} beat spectrum'.format(pickle_name)

    repet()
    est_bg, est_fg = repet.make_audio_signals()

    estimated = np.array([est_bg.get_channel(1), est_fg.get_channel(1)])
    true_srcs = np.array([true_bg.flatten(), true_fg.flatten()])

    mir_eval.separation.validate(true_srcs, estimated)
    bss_vals = mir_eval.separation.bss_eval_sources(true_srcs, estimated)


    sdr_dict = {'foreground': {'sdr': bss_vals[0][0], 'sir': bss_vals[1][0], 'sar': bss_vals[2][0]},
                'background': {'sdr': bss_vals[0][1], 'sir': bss_vals[1][1], 'sar': bss_vals[2][1]}}
    pickle.dump(sdr_dict, open(join(output_folder, '{}__sdrs.pick'.format(pickle_name)), 'wb'))
    print 'pickled {} sdrs'.format(pickle_name)




def get_wav_paths_in_folder(folder):
    return [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))
            and splitext(join(folder, f))[1] == '.wav']

if __name__ == '__main__':
    main()