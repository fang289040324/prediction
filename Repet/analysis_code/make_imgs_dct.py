import os
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.fftpack
import operator


def main():
    out = '../analysis_output/dct/'
    pickle_folder = '../pickles'


    for folder in [f for f in os.listdir(pickle_folder) if os.path.isdir(join(pickle_folder, f))]:
        beat_spec_name = join(pickle_folder, folder, folder + '__beat_spec.pick')
        sdrs_name = join(pickle_folder, folder, folder + '__sdrs.pick')

        name = os.path.basename(folder)
        name = splitext(name)[0]

        beat_spec = pickle.load(open(beat_spec_name, 'rb'))
        dct = scipy.fftpack.dct(beat_spec)

        sdr_vals = pickle.load(open(sdrs_name, 'rb'))


        plt.close('all')
        # plt.yscale('log')
        plt.plot(dct)
        plt.title('DCT beat spect {}, fore SDR = {}'.format(name, sdr_vals[0][0]))
        plt.xlabel('time')
        plt.savefig(join(out, '{}_dct.png'.format(folder)))




if __name__ == '__main__':
    main()
