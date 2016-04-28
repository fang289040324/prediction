import nussl
import os
import scipy.io.wavfile as wav
import numpy as np
from scipy.ndimage.interpolation import shift
import scipy.signal
from nussl import AudioMix
import pickle
import random
import itertools


def generate_dry_mixtures(src_dir, dest_dir, attn1, attn2):
    """
    Generates a set of mixtures from combinations of seed files in src_dir and saves them in dest_dir
    :return:
    """
    seeds = os.listdir(src_dir)

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in range(0, len(seeds)):
        for h in range(f+1, len(seeds)):
            if not os.path.exists(dest_dir+os.path.splitext(seeds[f])[0]+'+'+os.path.splitext(seeds[h])[0]+'.wav'):
                generate_mixture(src_dir+seeds[f], src_dir+seeds[h],
                                 dest_dir+os.path.splitext(seeds[f])[0]+'+'+os.path.splitext(seeds[h])[0]+'.wav',
                                 attn1, attn2)


def generate_impulse_responses(dest_dir):
    """
    Generates a set of impulse responses using nussl.Audiomix and saves them in dest_dir
    Generates a response for each combination of parameters in room_size_range and rcoef_range
    :return:
    """

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
            if not os.path.exists(dest_dir+str(i)+'_'+str(j)+'_IR.wav'):
                nussl.AudioSignal(audio_data_array=np.vstack((ir_1,ir_2)), sample_rate=44100)\
                    .write_audio_to_file(dest_dir+str(i)+'_'+str(j)+'_IR.wav', 44100)


def generate_reverb_mixtures(src_dir, ir_dir, dest_dir, iter_range):
    """
    Adds reverb from each IR in ir_dir to each mix in src_dir and saves them in dest_dir
    :return:
    """

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in os.listdir(src_dir):
        for h in os.listdir(ir_dir):
            generate_reverb(src_dir + f, ir_dir + h,
                            dest_dir+os.path.splitext(f)[0] + '_' + os.path.splitext(h)[0] +'.wav', iter_range)


def generate_pan_mixtures(src_dir, dest_dir, rel_attn_range):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in itertools.combinations(os.listdir(src_dir), 2):
        for a in rel_attn_range:
            generate_mixture(src_dir+f[0], src_dir+f[1],
                             dest_dir + os.path.splitext(f[0])[0] + '+' + os.path.splitext(f[1])[0] + '_'+str(a)+'.wav',
                             -a, a)


def generate_attenuation_varied_mixtures(seed_dir, dest_dir, attenuate_amounts):
    raise Exception('Depreciated')

    seeds = os.listdir(seed_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in range(0, len(seeds)):
        for h in range(f+1, len(seeds)):
            make_attn_varied_mixtures(seed_dir+seeds[f], seed_dir+seeds[h], dest_dir, attenuate_amounts)


def make_attn_varied_mixtures(seed1, seed2, dest_dir, attenuate_amounts):
    raise Exception('Depreciated')

    sr1, data1 = wav.read(seed1)
    if data1.dtype == np.dtype("int16"):
        data1 = data1 / float(np.iinfo(data1.dtype).max)

    sr2, data2 = wav.read(seed2)
    if data2.dtype == np.dtype("int16"):
        data2 = data2 / float(np.iinfo(data2.dtype).max)

    if sr1 != sr2:
        raise ValueError("Both sources muse have same sample rate")

    sample1 = data1[0:10 * sr1]
    sample2 = data2[0:10 * sr1]

    left = sample1 + sample2

    for attenuate_amount in attenuate_amounts:
        if os.path.exists(dest_dir + os.path.splitext(os.path.split(seed1)[1])[0] + '+' +
                                      os.path.splitext(os.path.split(seed2)[1])[0] + '_' + str(attenuate_amount) + '.wav'):
            continue
        right1 = np.copy(attenuate(sample1, .25))
        right2 = np.copy(attenuate(sample2, -.25))
        for i in xrange(0, len(right1)):
            right1[i] += random.random()*attenuate_amount * random.choice([-1,1])*right1[i]
        right = right1 + right2

        signal = np.vstack((left, right))
        scipy.io.wavfile.write(dest_dir+os.path.splitext(os.path.split(seed1)[1])[0]+'+'+os.path.splitext(os.path.split(seed2)[1])[0]+'_'+str(attenuate_amount)+'.wav', sr1, signal.T)


def generate_mixture(src1, src2, fname, attn1, attn2):
    """
    mixes 10 seconds of two sources of the same sample rate and saves them as fname
    :param src1:
    :param src2:
    :param fname:
    :param attn1:
    :param attn2:
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

    attn1 = float(attn1 + 1) / 2
    attn2 = float(attn2 + 1) / 2
    sample1 = data1[0:10 * sr1]
    sample2 = data2[0:10 * sr1]
    left = attenuate(sample1, attn1) + attenuate(sample2, attn2)
    right = attenuate(sample1, 1-attn1) + attenuate(sample2, 1-attn2)

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
    for i in xrange(0, iter_range+1):
        if i > 0:
            mix = add_reverb(prev_data.T, data_ir.T)
            prev_data = np.copy(mix).T
        else:
            mix = data.T
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


def main():
    #generate_dry_mixtures('audio/seed/', 'audio/mix/', .5, -.5)
    #generate_attenuation_varied_mixtures('audio/seed/', 'audio/mix_attn/', [3])
    #generate_impulse_responses('audio/IR/')
    #generate_reverb_mixtures('audio/mix/', 'audio/IR/downloaded/', 'audio/reverb_mix/', 5)
    #generate_pan_mixtures('audio/seed/', 'audio/pan_mix/', [0.0, 0.05, 0.125, .25, 0.375])
    generate_reverb_mixtures('audio/pan_mix/', 'audio/IR/downloaded/', 'audio/reverb_pan_mix_full/', 5)


def delay(data, delay):
    return shift(data, delay)


def attenuate(data, attenuation):
    return data * (1-attenuation)


def load_pickle(filename):
    if os.path.isfile(filename):
        return pickle.load(open(filename, 'rb'))
    return []


def save_pickle(obj, filename):
    model_file = open(filename, 'wb')
    pickle.dump(obj, model_file)
    model_file.close()

if __name__ == '__main__':
    main()
