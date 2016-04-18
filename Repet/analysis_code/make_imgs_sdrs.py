import Code.nussl
import os
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main():
    folder = 'training/audio/'
    out = 'training/'
    pickle_folder = 'training/pickles'

    means = []
    stds = []
    sdrs = []
    norm = []

    percent = 0.02

    for folder in [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]:
        beat_spec_name = join(pickle_folder, folder, folder + '__beat_spec.pick')
        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')

        beat_spec = pickle.load(open(beat_spec_name, 'rb'))
        top = beat_spec[np.argpartition(beat_spec, -int(len(beat_spec) * percent))][-int(len(beat_spec) * percent):]
        bottom = beat_spec[np.argpartition(beat_spec, -int(len(beat_spec) * percent))][:int(len(beat_spec) * percent)]
        mean_diff = np.mean(top) - np.mean(bottom)

        beat_spec_norm = beat_spec / max(beat_spec)
        top_norm = beat_spec_norm[np.argpartition(beat_spec_norm, -(len(beat_spec_norm) * percent))][
                   -int(len(beat_spec_norm) * percent):]
        bottom_norm = beat_spec_norm[np.argpartition(beat_spec_norm, (len(beat_spec_norm) * percent))][
                      :int(len(beat_spec_norm) * percent)]
        mean_diff_norm = np.mean(top_norm) - np.mean(bottom_norm)

        means.append(mean_diff)
        norm.append(mean_diff_norm)

        sdr_vals = pickle.load(open(sdrs_name, 'rb'))
        sdrs.append(sdr_vals[0][0])

    plt.yscale('log')
    plt.plot(sdrs, means, '.')
    plt.title('Beat spectrum diff btwn top 2% & bottom 2%')
    plt.xlabel('foreground SDR')
    plt.savefig(join(out, 'beat_spec_diff.png'))

    plt.close('all')
    plt.yscale('log')
    plt.plot(sdrs, norm, '.')
    plt.title('Beat spectrum normalized diff btwn top 2% & min 2%')
    plt.xlabel('foreground SDR')
    plt.savefig(join(out, 'beat_spec_mean2.png'))


if __name__ == '__main__':
    main()
