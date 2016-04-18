import os
from os.path import join, isfile, splitext
import nussl


def generate_repeating_files():
    base_folder = 'path/to/containing/folder/' # you might not need this if the script is in the same folder
    seed_folder = base_folder + 'seed/' # seed files go here
    output_folder = base_folder + 'ouput/'

    max_file_length = 10.0  # in seconds

    # loop through files in seed_folder
    for file in [f for f in os.listdir(seed_folder) if isfile(join(seed_folder, f))]:
        if splitext(file)[1] == '.wav':
            try:
                seed = nussl.AudioSignal(join(seed_folder, file))
                print('Read {}!'.format(file))
            except:
                print('Couldn\'t read {}'.format(file))
                continue

            create_looped_file(seed, max_file_length, file, output_folder)


def create_looped_file(audio_signal, max_file_length, seed_file_name, output_folder):
    max_samples = int(max_file_length * audio_signal.sample_rate)

    while len(audio_signal) < max_samples:
        audio_signal.concat(audio_signal)

    audio_signal.truncate_samples(max_samples)

    new_path = join(output_folder, splitext(seed_file_name)[0] + '_repeating' + splitext(seed_file_name)[1])
    audio_signal.write_audio_to_file(new_path, sample_rate=audio_signal.sample_rate, verbose=True)

if __name__ == '__main__':
    generate_repeating_files()
