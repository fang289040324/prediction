import mir_eval
import nussl
import os.path
import numpy as np

stft_swap_vals = np.arange(0.0, 0.425, 0.025)
noise_vals = np.arange(0.05, 0.50, 0.05)
speed_vals = np.arange(0.0, 0.55, 0.05)
none = [0.0]

# perturbation_info = {'stft_swap': (stft_swap_vals, '_'), 'none': (none, '_none_'), 'speed': (speed_vals, '_speed_')}
perturbation_info = { 'none': (none, '_none_')}

def main():
    stem = '/Users/ethanmanilow/Documents/School/Research/Predicting SDR values/prediction/Repet/'
    repet_out = stem + 'repet_output/'
    back_file_stem = stem + 'generated_audio/test_background/'
    fore_file_stem = stem + 'generated_audio/foreground/'
    mixt_file_stem = stem + 'generated_audio/test/'

    background_name = 'cowbell1'
    foreground_name = 'flute1'

    for name, (vals, string) in perturbation_info.iteritems():
        print '*' * 30, '\n', ' ' * 10, name, '\n', '*' * 30
        for n in vals:
            print '=' * 10, '> ', str(n), ' <', '=' * 10
            file_name_stem = foreground_name + '__' + background_name + string + str(n)
            full_file_name = mixt_file_stem + file_name_stem + '.wav'
            mixture = nussl.AudioSignal(full_file_name)
            r = nussl.Repet(mixture)
            # r.stft_params.window_type = nussl.WindowType.HANN
            r()
            r_back, r_fore = r.make_audio_signals()
            r_back.write_audio_to_file(repet_out + file_name_stem + '_back.wav', verbose=True)
            r_fore.write_audio_to_file(repet_out + file_name_stem + '_fore.wav', verbose=True)

            back_name = back_file_stem + background_name + '_' + name  + '_' + str(n) + '.wav'
            back = nussl.AudioSignal(back_name)
            back.truncate_seconds(10.0)
            if back.num_channels >= 2: back.to_mono(overwrite=True)

            fore_name = fore_file_stem + foreground_name  + '.wav'
            fore = nussl.AudioSignal(fore_name)
            fore.truncate_seconds(10.0)
            if fore.num_channels >= 2: fore.to_mono(overwrite=True)

            estimated = np.array([r_back.get_channel(1), r_fore.get_channel(1)])
            true_srcs = np.array([back.get_channel(1), fore.get_channel(1)])

            mir_eval.separation.validate(true_srcs, estimated)
            bss_val = mir_eval.separation.bss_eval_sources(true_srcs, estimated)
            print bss_val





    # for n in speed_vals:
    #     band_file_name = back_file_stem + 'band1_speed_' + str(n) + '.wav'
    #     a = nussl.AudioSignal(band_file_name)
    #
    #     mir_eval.separation.validate(back_perfect.audio_data, a.audio_data)
    #     sdr = mir_eval.separation.bss_eval_sources(back_perfect.audio_data, a.audio_data)
    #     print os.path.basename(band_file_name), '\t', sdr


if __name__ == '__main__':
    main()
