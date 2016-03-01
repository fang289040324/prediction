import nussl
import os
import scipy.io.wavfile as wav
import numpy as np
from scipy.ndimage.interpolation import shift
import scipy.signal
import matplotlib.pyplot as plt
from nussl import AudioMix
import pickle


def generate_dry_mixtures():
    """
    Generates a set of mixtures from combinations of seed files in src_dir and saves them in dest_dir
    :return:
    """
    src_dir = 'audio/seed/'
    dest_dir = 'audio/mix/'
    seeds = os.listdir(src_dir)

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in range(0, len(seeds)):
        for h in range(f+1, len(seeds)):
            if not os.path.exists(dest_dir+os.path.splitext(seeds[f])[0]+'_'+os.path.splitext(seeds[h])[0]+'.wav'):
                generate_mixture(src_dir+seeds[f], src_dir+seeds[h],
                                 dest_dir+os.path.splitext(seeds[f])[0]+'_'+os.path.splitext(seeds[h])[0]+'.wav')


def generate_impulse_responses():
    """
    Generates a set of impulse responses using nussl.Audiomix and saves them in dest_dir
    Generates a response for each combination of parameters in room_size_range and rcoef_range
    :return:
    """
    dest_dir = 'audio/IR/'

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    room_size_range = [.1, 1, 10]
    rcoef_range = [-1, 0, 1]
    for i in room_size_range:
        for j in rcoef_range:
            ir_1 = AudioMix.rir(np.array([i, i, i]), np.array([5.0, 5, 1]), np.array([2.5, 2.5, 1]), 1, j, 44100)
            ir_2 = AudioMix.rir(np.array([i, i, i]), np.array([4.75, 5, 1]), np.array([2.5, 2.5, 1]), 1, j, 44100)
            if ir_1.size < ir_2.size:
                ir_1.resize(ir_2.shape, refcheck=False)
            else:
                ir_2.resize(ir_1.shape, refcheck=False)
            if not os.path.exists(dest_dir+str(i)+'-'+str(j)+'_IR.wav'):
                nussl.AudioSignal(audio_data_array=np.vstack((ir_1,ir_2)), sample_rate=44100)\
                    .write_audio_to_file(dest_dir+str(i)+'-'+str(j)+'_IR.wav', 44100)


def generate_reverb_mixtures():
    """
    Adds reverb from each IR in ir_dir to each mix in src_dir and saves them in dest_dir
    :return:
    """
    src_dir = 'audio/mix/'
    ir_dir = 'audio/IR/'
    dest_dir = 'audio/outputs/'

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in os.listdir(src_dir):
        for h in os.listdir(ir_dir):
            generate_reverb(src_dir + f, ir_dir + h,
                            dest_dir+os.path.splitext(f)[0] + '-' + os.path.splitext(h)[0] +'.wav', [1, 2, 5, 10])


def run_duet(src_dir, dest_dir, pickle_dir, limit=0):
    """
    runs duet on all sources in src_dir and saves the result in dest_dir.
    :param src_dir:
    :param dest_dir:
    :param limit:
    :return:
    """
    j = 0
    for f in os.listdir(src_dir):
        sr, data = wav.read(src_dir+ f)
        if data.dtype == np.dtype("int16"):
            data = data / float(np.iinfo(data.dtype).max)

        sig = nussl.AudioSignal(audio_data_array=data.T, sample_rate=sr)

        duet = nussl.Duet(sig, 2)
        duet.run()

        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        if not os.path.exists(pickle_dir):
            os.mkdir(pickle_dir)

        save_pickle(duet, pickle_dir + os.path.splitext(f)[0]+'_duet.p')

        duet.plot(dest_dir + os.path.splitext(f)[0] + '_3d.png', True)

        output_name_stem = dest_dir + os.path.splitext(f)[0] + '_duet_source'
        i = 1
        for s in duet.make_audio_signals():
            output_file_name = output_name_stem + str(i) + '.wav'
            s.write_audio_to_file(output_file_name, sample_rate=sr)
            i += 1
        j += 1
        if j == limit:
            break


def generate_mixture(src1, src2, fname):
    """
    mixes two sources of the same sample rate and saves them as fname
    :param src1:
    :param src2:
    :param fname:
    :return:
    """
    sr1, data1 = wav.read(src1)
    if data1.dtype == np.dtype("int16"):
        data1 = data1 / float(np.iinfo(data1.dtype).max)

    sr2, data2 = wav.read(src2)
    if data2.dtype == np.dtype("int16"):
        data2 = data2 / float(np.iinfo(data2.dtype).max)

    if sr1 != sr2:
        raise ValueError("Both sources muse have same sample rate")

    sample1 = data1[0:10 * sr1]
    sample2 = data2[0:10 * sr1]
    left = sample1 + sample2
    right = attenuate(sample1, .25) + attenuate(sample2, -.25)

    signal = np.vstack((left, right))
    scipy.io.wavfile.write(fname, sr1, signal.T)


def generate_reverb(signal, reverb, fname, iter_range):
    """
    Adds reverb from the path reverb to the data in the path signal and saves it as fname. Applies reverb iteratively over
    iter_range
    :param signal:
    :param reverb:
    :param fname:
    :param iter_range:
    :return:
    """
    sr, data = wav.read(signal)
    if data.dtype == np.dtype("int16"):
        data = data / float(np.iinfo(data.dtype).max)


    sr_ir, data_ir = wav.read(reverb)
    if data_ir.dtype == np.dtype("int16"):
        data_ir = data_ir / float(np.iinfo(data_ir.dtype).max)

    if sr_ir != sr:
        raise ValueError("Impulse Response must have same sample rate as signal")

    prev_data = data
    for i in iter_range:
        mix = add_reverb(prev_data.T, data_ir.T)
        prev_data = np.copy(mix).T
        if not os.path.exists(os.path.splitext(fname)[0]+'-'+str(i)+'.wav'):
            scipy.io.wavfile.write(os.path.splitext(fname)[0]+'-'+str(i)+'.wav', sr, mix.T)


def add_reverb(signal, impulse_response):
    """
    Adds reverb from impulse_response to the signal using convolution in the frequency domain
    :param signal: a multi-channel signal
    :param impulse_response: a multi-channel impulse response function
    :return:
    """
    outl = scipy.signal.fftconvolve(signal[0], impulse_response[0])
    outr = scipy.signal.fftconvolve(signal[1], impulse_response[1])
    combo = np.vstack((outl, outr))
    return combo / abs(combo).max()


def calculate_sdrs(pickle_dir):
    pickles = os.listdir(pickle_dir)

    for f in pickles:
        duet = load_pickle(pickle_dir + f)  # type: nussl.Duet
        separated_srcs = duet.separated_sources
        real_src_paths = f.split('_')


def main():
    #generate_dry_mixtures()
    #generate_impulse_responses()
    #generate_reverb_mixtures()

    if not os.path.exists('Output/'):
        os.mkdir('Output/')
    if not os.path.exists('Output/pickle/'):
        os.mkdir('Output/pickle/')
    run_duet('audio/mix/', 'Output/mix/', 'Output/pickle/mix/')
    #run_duet('audio/outputs/', 'Output/reverb/', 'Output/pickle/reverb/')
    #run_duet('audio/tests/', 'Output/test/', 'Output/pickle/tests/')

    calculate_sdrs('Output/pickle/mix/')


def delay(data, delay):
    return shift(data, delay)


def attenuate(data, attenuation):
    return data * (1-attenuation)


def load_pickle(filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))
    return []


def save_pickle(object, filename):
    model_file = open(filename, 'wb')
    pickle.dump(object, model_file)
    model_file.close()



def plot_stereo(signal, fs):
    """
    Plots a stereo signal using matplotlib
    :param signal: signal data
    :param fs: sample rate
    :return:
    """
    t = np.linspace(0, signal[0].size / fs, num=signal[0].size)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.grid(True)
    ax1.axhline(0, color='black', lw=2)
    ax1.plot(t,signal[0])

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=2)
    ax2.plot(t,signal[1])

    plt.show()

if __name__ == '__main__':
    main()
