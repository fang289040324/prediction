import nussl
import os
import scipy.io.wavfile as wav
import numpy as np
from scipy.ndimage.interpolation import shift
import scipy.signal
import matplotlib.pyplot as plt
from nussl import AudioMix
import matlab_wrapper


def generate_dry_mixtures():
    src_dir = 'audio/seed/'
    dest_dir = 'audio/mix/'
    seeds = os.listdir(src_dir)

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in range(0, len(seeds)):
        for h in range(f, len(seeds)):
            if not os.path.exists(dest_dir+os.path.splitext(seeds[f])[0]+'_'+os.path.splitext(seeds[h])[0]+'.wav'):
                generate_mixture(src_dir+seeds[f], src_dir+seeds[h],
                                 dest_dir+os.path.splitext(seeds[f])[0]+'_'+os.path.splitext(seeds[h])[0]+'.wav')


def generate_impulse_responses():
    dest_dir = 'audio/IR/'

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    room_size_range = [1, 10, 100]
    rcoef_range = [-1, 0, 1]
    for i in room_size_range:
        for j in rcoef_range:
            ir_1 = AudioMix.rir(np.array([i, i, 10]), np.array([5.0, 5, 1]), np.array([2.5, 2.5, 1]), 1, j, 44100)
            ir_2 = AudioMix.rir(np.array([i, i, 10]), np.array([4.75, 5, 1]), np.array([2.5, 2.5, 1]), 1, j, 44100)
            if ir_1.size < ir_2.size:
                ir_1.resize(ir_2.shape, refcheck=False)
            else:
                ir_2.resize(ir_1.shape, refcheck=False)
            if not os.path.exists(dest_dir+str(i)+'_'+str(j)+'_IR.wav'):
                nussl.AudioSignal(audio_data_array=np.vstack((ir_1,ir_2)), sample_rate=44100)\
                    .write_audio_to_file(dest_dir+str(i)+'_'+str(j)+'_IR.wav', 44100)


def generate_reverb_mixtures():
    src_dir = 'audio/mix/'
    ir_dir = 'audio/IR/'
    dest_dir = 'audio/outputs/'

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in os.listdir(src_dir):
        for h in os.listdir(ir_dir):
            generate_reverb(src_dir + f, ir_dir + h,
                            dest_dir+os.path.splitext(f)[0] + '_' + os.path.splitext(h)[0] +'.wav', [1, 2, 5, 10, 30])


def generate_duet_histograms(src_dir, dest_dir, limit=0):
    j = 0
    for f in os.listdir(src_dir):
        sr, data = wav.read(src_dir+ f)
        if data.dtype == np.dtype("int16"):
            data = data / float(np.iinfo(data.dtype).max)

        sig = nussl.AudioSignal(audio_data_array=data.T)

        duet = nussl.Duet(sig, sample_rate=sr)
        duet.run()

        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        duet.plot(dest_dir + os.path.splitext(f)[0] + '_2d.png')
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
        if not os.path.exists(os.path.splitext(fname)[0]+'_'+str(i)+'.wav'):
            scipy.io.wavfile.write(os.path.splitext(fname)[0]+'_'+str(i)+'.wav', sr, mix.T)


def plot_stereo(signal, fs):
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


def delay(data, delay):
    return shift(data, delay)


def attenuate(data, attenuation):
    return data * (1-attenuation)


def add_reverb(signal, impulse_response):
    if len(impulse_response.shape) > 1 :
        outl = scipy.signal.fftconvolve(signal[0], impulse_response[0])
        outr = scipy.signal.fftconvolve(signal[1], impulse_response[1])
    else:
        outl = scipy.signal.fftconvolve(signal[0], impulse_response)
        outr = scipy.signal.fftconvolve(signal[1], impulse_response)
    combo = np.vstack((outl, outr))
    return combo / abs(combo).max()


def main():
    generate_dry_mixtures()
    generate_impulse_responses()
    #generate_reverb_mixtures()

    if not os.path.exists('Output/'):
        os.mkdir('Output/')
    generate_duet_histograms('audio/mix/', 'Output/mix/')
    #generate_duet_histograms('audio/outputs/', 'Output/reverb/')
    generate_duet_histograms('audio/tests/', 'Output/test/')

    matlab = matlab_wrapper.MatlabSession();



if __name__ == '__main__':
    main()
