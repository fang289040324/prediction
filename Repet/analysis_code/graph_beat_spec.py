import nussl
import os
from os.path import join, splitext
import numpy as np

if __name__ == '__main__':
    background_folder = '../generated_audio/test_background/'
    out_folder = '../analysis_output/beat_spec/'
    background_paths = [join(background_folder, f) for f in os.listdir(background_folder) if '_swap_stft_0.4' in f]
    n = 8192.0
    curve = np.arange(0.0, np.pi/2, np.pi/n/2)
    # curve = np.cos(curve)
    curve = curve[::-1]
    suffix = '_bad_beat_spec_'

    for path in background_paths:
        sig = nussl.AudioSignal(path)
        # sig.audio_data *= 100
        # sig.plot_spectrogram(join(out_folder, splitext(sig.file_name)[0]))
        # sig.audio_data[:, -n:] *= curve
        # sig.write_audio_to_file(join(out_folder, splitext(sig.file_name)[0] + suffix + '.wav'))
        r = nussl.Repet(sig)
        name = join(out_folder, splitext(sig.file_name)[0] + suffix + '.png')
        r.plot(name, title='')
        print 'plotted ', name
        # break