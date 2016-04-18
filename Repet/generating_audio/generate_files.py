import copy
import os
import random
import sys

import numpy as np

print os.path.dirname(sys.executable)
from os.path import join, isfile, splitext
import nussl
import librosa


def generate_random_files_spectral_swap():
    base_folder = '../generated_audio/'
    seedInputFolder = base_folder + 'seed/'
    bkgndOutputFolder = base_folder + 'background/'

    foregroundInputFolder = base_folder + 'foreground/'
    mixtureOutputFolder = base_folder + 'training/'

    maxFileLength = 10.0  # in seconds
    perturbations = (swap_stft_values, add_noise_time)

    # make backgrounds
    for file in [f for f in os.listdir(seedInputFolder) if isfile(join(seedInputFolder, f))]:
        if splitext(file)[1] == '.wav':
            try:
                seed = nussl.AudioSignal(join(seedInputFolder, file))
                print('Read {}!'.format(file))
                if seed.num_channels != 1:
                    print'File {} not mono!'.format(seed.file_name)
                    seed.audio_data = seed.get_channel(1)
            except:
                print('Couldn\'t read {}'.format(file))
                continue

            if seed.sample_rate != nussl.Constants.DEFAULT_SAMPLE_RATE:
                print('Skipping {0} because its sample rate isn\'t {1}'.format(
                    seed.file_name, nussl.Constants.DEFAULT_SAMPLE_RATE))
                continue

            create_looped_file_speed_change(seed, maxFileLength, file, bkgndOutputFolder)

            audioSig = create_pure_looped_file(seed, maxFileLength, file, bkgndOutputFolder)
            for perturbation in perturbations:
                perturbation(audioSig, bkgndOutputFolder, maxFileLength)

    # add backgrounds and foregrounds
    for fgndFile in [f for f in os.listdir(foregroundInputFolder) if isfile(join(foregroundInputFolder, f))]:
        if splitext(fgndFile)[1] == '.wav':
            try:
                fgnd = nussl.AudioSignal(join(foregroundInputFolder, fgndFile))
                print('Read {}!'.format(fgndFile))
                if fgnd.num_channels != 1:
                        print'File {} not mono!'.format(fgnd.file_name)
                        fgnd.audio_data = fgnd.get_channel(1)
            except:
                print('Couldn\'t read {}'.format(fgndFile))
                continue

            if fgnd.sample_rate != nussl.Constants.DEFAULT_SAMPLE_RATE:
                print('Skipping {0} because its sample rate isn\'t {1}'.format(
                    fgnd.file_name, nussl.Constants.DEFAULT_SAMPLE_RATE))
                continue

            if len(fgnd) > int(maxFileLength * fgnd.sample_rate):
                fgnd.truncate_seconds(int(maxFileLength))

            if len(fgnd) < int(maxFileLength * fgnd.sample_rate):
                # z = np.zeros((int(maxFileLength * fgnd.sample_rate) - len(fgnd)))
                # z = z[np.newaxis, :]
                # fgnd.audio_data = np.concatenate((fgnd.audio_data, z))
                fgnd.zero_pad(0, (int(maxFileLength * fgnd.sample_rate) - len(fgnd)))
                fgnd.truncate_seconds(int(maxFileLength))
                fgnd.write_audio_to_file(join(foregroundInputFolder, fgndFile), verbose=True)

            for bkgdFile in next(os.walk(bkgndOutputFolder))[2]:
                if splitext(bkgdFile)[1] != '.wav':
                    continue

                combfile_name = splitext(fgndFile)[0] + '__' + bkgdFile
                combPath = join(mixtureOutputFolder, combfile_name)

                if os.path.exists(combPath):
                    print 'Skipping {0} & {1} because they\'re already combined :)'.format(fgndFile, bkgdFile)
                    continue

                path = join(bkgndOutputFolder, bkgdFile)
                try:
                    bkgd = nussl.AudioSignal(path)
                    combined = bkgd + fgnd
                except Exception, e:
                    print('Couldn\'t read {}'.format(path))
                    continue

                combined.write_audio_to_file(combPath, sample_rate=nussl.Constants.DEFAULT_SAMPLE_RATE, verbose=True)


    # split into training and test
    # Going to hold off on this until we're more confident about things...
    # test_folder = 'mixture_test/'
    # percent = 0.1
    # all_examples = [f for f in os.listdir(mixtureOutputFolder) if isfile(join(mixtureOutputFolder, f))]
    # indices = random.sample(range(len(all_examples)), int(len(all_examples) * percent))
    # for i in indices:
    #     cur = join(mixtureOutputFolder, all_examples[i])
    #     dest = join(test_folder, all_examples[i])
    #     os.rename(cur, dest)


def create_pure_looped_file(audioSignal, maxFileLength, fileName, outputFolder):
    maxSamples = int(maxFileLength * audioSignal.sample_rate)

    while len(audioSignal) < maxSamples:
        audioSignal.concat(audioSignal)

    audioSignal.truncate_samples(maxSamples)

    newPath = join(outputFolder, splitext(fileName)[0] + '_0.0' + splitext(fileName)[1])
    audioSignal.write_audio_to_file(newPath)

    return audioSignal


def swap_stft_values(audioSig, outputFolder, max_duration):
    step = 0.025
    min = step
    max = 0.4 + step

    audioSig.stft()

    path = splitext(audioSig.file_name.split(os.sep)[-1])

    I, J, ch = audioSig.stft_data.shape
    for num in np.arange(min, max, step):
        fileName = path[0] + '_' + str(num) + path[1]
        outputPath = join(outputFolder, fileName)

        if os.path.exists(outputPath):
            # print 'Skipping {} because it exists :)'.format(fileName)
            # continue
            print 'redoing {}'.format(fileName)

        audioSig_copy = copy.copy(audioSig)
        audioSig_copy.stft()

        for n in range(int(audioSig_copy.stft_data.size * num)):
            i1 = random.randint(0, I - 1)
            j1 = random.randint(0, J - 1)
            i2 = random.randint(0, I - 1)
            j2 = random.randint(0, J - 1)
            audioSig_copy.stft_data[i1, j1], audioSig_copy.stft_data[i2, j2] = \
                audioSig_copy.stft_data[i2, j2], audioSig_copy.stft_data[i1, j1]

        audioSig_copy.istft()
        audioSig_copy.truncate_seconds(max_duration)
        audioSig_copy.write_audio_to_file(outputPath, verbose=True)


def add_noise_time(audio_signal, output_folder, max_duration):
    step = 0.05
    min = step
    max = 0.5 + step

    path = splitext(audio_signal.file_name.split(os.sep)[-1])

    for num in np.arange(min, max, step):
        file_name = path[0] + '_noise_time_' + str(num) + path[1]
        output_path = join(output_folder, file_name)

        if os.path.exists(output_path):
            print 'Skipping {} because it exists :)'.format(file_name)
            continue

        audio_signal_copy = copy.copy(audio_signal)
        audio_signal_copy.audio_data += num * np.random.rand(audio_signal_copy.audio_data.shape[1])
        audio_signal_copy.write_audio_to_file(output_path, verbose=True)


def create_looped_file_speed_change(audio_signal, maxFileLength, fileName, outputFolder):
    maxSamples = int(maxFileLength * audio_signal.sample_rate)
    audio_signal_copy = copy.copy(audio_signal)

    step = 0.05
    min = 0.3
    max = 1.0 + step

    for num in np.arange(min, max, step):
        output_path = join(outputFolder, splitext(fileName)[0] + '_speed_' + str(num) + splitext(fileName)[1])
        if os.path.exists(output_path):
            print 'Skipping speed variation of {} because it exists :)'.format(fileName)
            continue

        audio_signal_speed = copy.copy(audio_signal)
        output = np.array(())
        y = audio_signal_speed.get_channel(1)

        while len(output) < maxSamples:
            rand = np.random.random()
            n = 1 / ( num * (rand * 2. - 1.) + 1.)
            y_eff = librosa.effects.time_stretch(y, n)
            output = np.hstack((output, y_eff))

        audio_signal_copy.audio_data = output
        audio_signal_copy.truncate_samples(maxSamples)


        audio_signal_copy.write_audio_to_file(output_path, verbose=True)




if __name__ == '__main__':
    generate_random_files_spectral_swap()
