import copy
import os
import numpy as np
import operator
import matplotlib.pyplot as plt
from os.path import join, isfile, splitext
import nussl
import librosa
import mir_eval

number_of_seed_files = 2
speed = 'speed'
stft_nudge = 'stft_nudge'
noise_time = 'noise_time'
swap_stft = 'swap_stft'
zero_stft = 'zero_stft'
noise_stft = 'noise_stft'
zero_stft_nr = 'zero_nr_stft'
noise_stft_nr = 'noise_nr_stft'
none = 'none'
global_perturbations = [none, swap_stft, speed, zero_stft, noise_stft, zero_stft_nr, noise_stft_nr]
volume_perturbations = [zero_stft, zero_stft_nr, noise_stft, noise_stft_nr]
max_file_length = 10.0  # in seconds

# folders
base_folder = '../generated_audio/'
seed_input_folder = base_folder + 'seed/'
bkgnd_output_folder = base_folder + 'test_background/'

foreground_input_folder = base_folder + 'foreground/'
mixture_output_folder = base_folder + 'test/'
out = 'graphs/'


def generate_all_files():
    bg_perturbations = (swap_stft_values, zero_stft_values_no_replacement)
    # bg_perturbations = (add_noise_stft_values_no_replacement,)

    background_seeds = get_wav_files_in_folder(seed_input_folder)
    foreground_files = get_wav_files_in_folder(foreground_input_folder)

    # background_seeds = [background_seeds[i] for i in np.random.permutation(len(background_seeds))][0:number_of_seed_files]
    # foreground_files = [foreground_files[i] for i in np.random.permutation(len(foreground_files))][0:number_of_seed_files]

    # make backgrounds
    for file in background_seeds:
        seed = read_file(join(seed_input_folder, file))
        if seed is None:
            continue

        create_looped_file_speed_change(seed, max_file_length, file, bkgnd_output_folder)
        # create_looped_file_stft_nudge(seed, max_file_length, file, bkgnd_output_folder)

        pure_looped_file = create_pure_looped_file(seed, max_file_length, file, bkgnd_output_folder)
        for perturbation in bg_perturbations:
            perturbation(pure_looped_file, bkgnd_output_folder, max_file_length)

    # add backgrounds and foregrounds
    for fgnd_file in foreground_files:

        fgnd = read_file(join(foreground_input_folder, fgnd_file))
        if fgnd is None:
            continue

        if len(fgnd) > int(max_file_length * fgnd.sample_rate):
            fgnd.truncate_seconds(int(max_file_length))

        if len(fgnd) < int(max_file_length * fgnd.sample_rate):
            fgnd.zero_pad(0, (int(max_file_length * fgnd.sample_rate) - len(fgnd)))
            fgnd.truncate_seconds(int(max_file_length))
            fgnd.write_audio_to_file(join(foreground_input_folder, fgnd_file), verbose=True)

        fg_rms = fgnd.rms()

        for bkgd_file in next(os.walk(bkgnd_output_folder))[2]:
            if splitext(bkgd_file)[1] != '.wav':
                continue

            combined_file_name = splitext(fgnd_file)[0] + '__' + bkgd_file
            combined_path = join(mixture_output_folder, combined_file_name)

            if os.path.exists(combined_path):
                print 'Skipping {0} & {1} because they\'re already combined :)'.format(fgnd_file, bkgd_file)
                # continue

            path = join(bkgnd_output_folder, bkgd_file)
            try:
                bkgd = nussl.AudioSignal(path)
                bg_rms = bkgd.rms()

                perturb_type = [p for p in global_perturbations if p in bkgd.file_name][0]
                if perturb_type not in volume_perturbations:
                    if fg_rms > bg_rms:
                        fgnd.audio_data *= bg_rms / fg_rms
                    else:
                        bkgd.audio_data *= fg_rms / bg_rms

                combined = bkgd + fgnd
            except Exception, e:
                print('Couldn\'t read {}'.format(path))
                continue

            combined.write_audio_to_file(combined_path, sample_rate=nussl.constants.DEFAULT_SAMPLE_RATE, verbose=True)


def read_file(path_to_file):
    if splitext(path_to_file)[1] != '.wav':
        return None

    try:
        file = nussl.AudioSignal(path_to_file)
        print('Read {}!'.format(file.file_name))
        if file.num_channels != 1:
            print'File {} not mono!'.format(file.file_name)
            file.to_mono(True)
    except:
        print('Couldn\'t read {}'.format(path_to_file))
        return None

    if file.sample_rate != nussl.constants.DEFAULT_SAMPLE_RATE:
        print('Skipping {0} because its sample rate isn\'t {1}'.format(
            file.file_name, nussl.constants.DEFAULT_SAMPLE_RATE))
        return None

    return file


def create_pure_looped_file(audio_signal, max_file_length, file_name, output_folder):
    max_samples = int(max_file_length * audio_signal.sample_rate)

    while len(audio_signal) < max_samples:
        audio_signal.concat(audio_signal)

    audio_signal.truncate_samples(max_samples)

    newPath = join(output_folder, splitext(file_name)[0] + '_' + none + '_0.0' + splitext(file_name)[1])
    audio_signal.to_mono()
    audio_signal.write_audio_to_file(newPath)

    return audio_signal


def swap_stft_values(audio_signal, output_folder, max_duration):
    step = 0.025
    min = step
    max = 0.4 + step

    audio_signal.stft()

    path = splitext(audio_signal.file_name.split(os.sep)[-1])

    I, J, ch = audio_signal.stft_data.shape
    for num in np.arange(min, max, step):
        file_name = path[0] + '_' + swap_stft + '_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            continue
            # print 'redoing {}'.format(file_name)

        audio_sig_copy = copy.copy(audio_signal)
        stft = audio_sig_copy.stft().flatten()

        perm_i = np.random.permutation(int(len(stft) * num))
        if len(perm_i) % 2 != 0:
            perm_i = np.insert(perm_i, np.random.randint(0, len(perm_i)), np.random.randint(0, len(perm_i)))
        perm_i = perm_i.reshape((len(perm_i)/2, 2))

        for n, m in perm_i:
            stft[n], stft[m] = stft[m], stft[n]

        audio_sig_copy.stft_data = stft.reshape(audio_signal.stft_data.shape)
        audio_sig_copy.istft()
        audio_sig_copy.truncate_seconds(max_duration)
        audio_sig_copy.write_audio_to_file(output_path, verbose=True)


def add_noise_time(audio_signal, output_folder, max_duration):
    step = 0.05
    min = step
    max = 0.5 + step

    path = splitext(audio_signal.file_name.split(os.sep)[-1])

    for num in np.arange(min, max, step):
        file_name = path[0] + '_' + noise_time + '_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            continue

        audio_signal_copy = copy.copy(audio_signal)
        audio_signal_copy.audio_data += num * np.random.rand(audio_signal_copy.audio_data.shape[1])
        audio_signal_copy.write_audio_to_file(output_path, verbose=True)


def zero_stft_values(audio_signal, output_folder, max_duration):
    step = 0.025
    min = step
    max = 0.8 + step

    audio_signal.stft()

    path = splitext(audio_signal.file_name.split(os.sep)[-1])

    for num in np.arange(min, max, step):
        file_name = path[0] + '_' + zero_stft + '_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            # continue
            print 'redoing {}'.format(file_name)

        audio_sig_copy = copy.copy(audio_signal)
        stft = audio_sig_copy.stft().flatten()

        perm_i = np.random.permutation(int(len(stft) * num))
        for n in perm_i:
            stft[n] = 0.0

        audio_sig_copy.stft_data = stft.reshape(audio_signal.stft_data.shape)
        audio_sig_copy.istft()
        audio_sig_copy.truncate_seconds(max_duration)
        audio_sig_copy.write_audio_to_file(output_path, verbose=True)

def zero_stft_values_no_replacement(audio_signal, output_folder, max_duration):
    step = 0.025
    min = step
    max = 1.0
    values = np.arange(min, max, step)

    path = splitext(audio_signal.file_name.split(os.sep)[-1])
    audio_sig_copy = copy.copy(audio_signal)
    stft = audio_sig_copy.stft().flatten()
    perm = np.random.permutation(len(stft))

    for num in values:
        file_name = path[0] + '_' + zero_stft_nr + '_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            # continue
            print 'redoing {}'.format(file_name)

        beg = int((num-step) * len(perm))
        end = int(num * len(perm))
        perm_i = perm[beg:end]
        for n in perm_i:
            stft[n] = 0.0

        audio_sig_copy.stft_data = stft.reshape(audio_sig_copy.stft_data.shape)
        audio_sig_copy.istft()
        audio_sig_copy.truncate_seconds(max_duration)
        audio_sig_copy.write_audio_to_file(output_path, verbose=True)

def add_noise_stft_values(audio_signal, output_folder, max_duration):
    step = 0.025
    min = step
    max = 0.8 + step

    audio_signal.stft()

    path = splitext(audio_signal.file_name.split(os.sep)[-1])

    for num in np.arange(min, max, step):
        file_name = path[0] + '_' + noise_stft + '_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            continue
            # print 'redoing {}'.format(file_name)

        audio_sig_copy = copy.copy(audio_signal)
        stft = audio_sig_copy.stft().flatten()

        perm_i = np.random.permutation(int(len(stft) * num))
        for n in perm_i:
            stft[n] *= np.random.random()

        audio_sig_copy.stft_data = stft.reshape(audio_signal.stft_data.shape)
        audio_sig_copy.istft()
        audio_sig_copy.truncate_seconds(max_duration)
        audio_sig_copy.write_audio_to_file(output_path, verbose=True)

def add_noise_stft_values_no_replacement(audio_signal, output_folder, max_duration):
    step = 0.025
    min = step
    max = 0.8 + step

    audio_signal.stft()

    path = splitext(audio_signal.file_name.split(os.sep)[-1])
    audio_sig_copy = copy.copy(audio_signal)
    stft = audio_sig_copy.stft().flatten()
    perm = np.random.permutation(len(stft))

    for num in np.arange(min, max, step):
        file_name = path[0] + '_' + noise_stft_nr + '_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            continue
            # print 'redoing {}'.format(file_name)

        beg = int((num - step) * len(perm))
        end = int(num * len(perm))
        perm_i = perm[beg:end]
        for n in perm_i:
            stft[n] *= np.random.random()

        audio_sig_copy.stft_data = stft.reshape(audio_signal.stft_data.shape)
        audio_sig_copy.istft()
        audio_sig_copy.truncate_seconds(max_duration)
        audio_sig_copy.write_audio_to_file(output_path, verbose=True)

def create_looped_file_stft_nudge(audio_signal, max_file_length, file_name, output_folder):
    audio_signal_copy = copy.copy(audio_signal)
    max_audio_sig = create_pure_looped_file(audio_signal, max_file_length, file_name, output_folder)
    max_audio_sig.stft()
    max_stft_bins = len(max_audio_sig.stft_data[max_audio_sig._STFT_LEN])

    min = 0
    max = 10

    audio_signal_copy.stft()

    for pitch_range in range(1, max+1):
        output_path = join(output_folder,
                           splitext(file_name)[0] + '_' + stft_nudge + '_' + str(pitch_range) + splitext(file_name)[1])
        if os.path.exists(output_path):
            print 'Skipping stft_nudge of {} because it exists :)'.format(file_name)
            continue

        cur_audio_sig_copy = copy.copy(audio_signal_copy)
        stft_data = np.copy(cur_audio_sig_copy.stft_data)

        while len(stft_data[cur_audio_sig_copy._STFT_LEN]) < max_stft_bins:
            nudge = np.random.randint(pitch_range+1)

            shifted = np.lib.pad(stft_data, nudge, 'wrap')

            if np.random.randint(2) == 1:
                shifted = shifted





def create_looped_file_speed_change(audio_signal, max_file_length, file_name, output_folder):
    max_samples = int(max_file_length * audio_signal.sample_rate)
    audio_signal_copy = copy.copy(audio_signal)

    step = 0.025
    min = 0.05
    max = 0.5 + step

    for num in np.arange(min, max, step):
        output_path = join(output_folder,
                           splitext(file_name)[0] + '_' + speed + '_' + str(num) + splitext(file_name)[1])
        if os.path.exists(output_path):
            print 'Skipping speed variation of {} because it exists :)'.format(file_name)
            continue

        audio_signal_speed = copy.copy(audio_signal)
        output = np.array(())
        y = audio_signal_speed.get_channel(1)

        while len(output) < max_samples:
            rand = np.random.random()
            n = 1 / (num * (rand * 2. - 1.) + 1.)
            y_eff = librosa.effects.time_stretch(y, n)
            output = np.hstack((output, y_eff))

        audio_signal_copy.audio_data = output
        audio_signal_copy.truncate_samples(max_samples)

        audio_signal_copy.write_audio_to_file(output_path, verbose=True)


def run_repet_and_graph_sdrs(mix_folder, bg_folder, fg_folder):
    mixture_paths = get_wav_paths_in_folder(mix_folder)
    sdr_dict = {p:{} for p in global_perturbations}
    backgrounds = True

    for mixture_path in mixture_paths:
        print 'Running REPET for {}'.format(mixture_path)
        mixture = nussl.AudioSignal(mixture_path)

        fg_stem, bg_name = mixture.file_name.split('__')
        bg_stem = bg_name[:bg_name.find('_')]
        perturb_val = float(os.path.splitext(bg_name[bg_name.rfind('_') + 1:])[0])
        fg_name = fg_stem + '.wav'
        background = nussl.AudioSignal(os.path.join(bg_folder, bg_name))
        foreground = nussl.AudioSignal(os.path.join(fg_folder, fg_name))

        perturb_type = [p for p in global_perturbations if p in bg_name][0]

        r = nussl.Repet(mixture)
        r()
        est_bg, est_fg = r.make_audio_signals()

        # truncate
        background.truncate_seconds(max_file_length)
        foreground.truncate_seconds(max_file_length)
        est_bg.truncate_seconds(max_file_length)
        est_fg.truncate_seconds(max_file_length)

        if perturb_val not in sdr_dict[perturb_type]:
            # sdr_dict[perturb_type]['name'] = fg_stem + ' ' + bg_stem
            sdr_dict[perturb_type][perturb_val] = []

        if backgrounds:
            mir_eval.separation.validate(background.audio_data, est_bg.audio_data)
            sdr_vals = zip(*mir_eval.separation.bss_eval_sources(background.audio_data, est_bg.audio_data))[0]
        else:
            mir_eval.separation.validate(foreground.audio_data, est_fg.audio_data)
            sdr_vals = zip(*mir_eval.separation.bss_eval_sources(foreground.audio_data, est_fg.audio_data))[0]

        sdr = sdr_vals[0]
        sdr_dict[perturb_type][perturb_val].append(sdr)

    for p_name, dict in sdr_dict.iteritems():
        if none in p_name or dict == {}:
            continue

        dict[0.0] = sdr_dict[none].values()[0]
        # name = dict.pop('name')
        sorted_sdrs = sorted(dict.items(), key=operator.itemgetter(0))
        sdr_values = [sorted_sdrs[v][1] for v in range(len(sorted_sdrs))]
        sdr_coefs = [str(sorted_sdrs[k][0]) for k in range(len(sorted_sdrs))]

        plt.close('all')
        plt.plot(sdr_values)
        plt.xticks(range(len(sdr_values)+1), sdr_coefs, rotation='vertical')
        plt.title('repetition perturbation coef vs. background SDR {}'.format(p_name))
        plt.xlabel('perturbation coefficient')
        plt.ylabel('SDR (dB)')
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(out + 'dataset_correlation_background_{}.png'.format(p_name))

def make_combined_path(folder, fg_name, bg_name):
    name = fg_name + '__' + bg_name + '.wav'
    return os.path.join(folder, name)


def make_perturbation_name(bg_stem, perturbation, number, no_wav=False):
    name = bg_stem + '_' + perturbation + '_' + number
    name += '.wav' if no_wav else ''
    return name


def get_wav_files_in_folder(folder):
    return [f for f in os.listdir(folder) if isfile(join(folder, f))
            and splitext(join(folder, f))[1] == '.wav']


def get_wav_paths_in_folder(folder):
    return [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))
            and splitext(join(folder, f))[1] == '.wav']


if __name__ == '__main__':
    generate_all_files()
    run_repet_and_graph_sdrs(mixture_output_folder, bkgnd_output_folder, foreground_input_folder)
