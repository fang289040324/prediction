import nussl
import os
import scipy.io.wavfile as wav
import numpy as np
from scipy.ndimage.interpolation import shift
import scipy.signal
import matplotlib.pyplot as plt
from nussl import AudioMix
import pickle
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bss_eval'))
import mir_eval
from sklearn import mixture
import csv

def generate_dry_mixtures(src_dir, dest_dir):
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
                                 dest_dir+os.path.splitext(seeds[f])[0]+'+'+os.path.splitext(seeds[h])[0]+'.wav')


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


def generate_attenuation_varied_mixtures(seed_dir, dest_dir, attenuate_amounts):
    seeds = os.listdir(seed_dir)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for f in range(0, len(seeds)):
        for h in range(f+1, len(seeds)):
            make_attn_varied_mixtures(seed_dir+seeds[f], seed_dir+seeds[h], dest_dir, attenuate_amounts)


def make_attn_varied_mixtures(seed1, seed2, dest_dir, attenuate_amounts):
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
    right = attenuate(sample1, .5) + attenuate(sample2, -.5)

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


def calculate_sdrs(extracted_src_list, original_src_paths):

    signal_len = len(extracted_src_list[0])
    num_srcs = len(extracted_src_list)

    reference_sources = np.zeros((num_srcs, signal_len))
    for i in xrange(0, len(original_src_paths)):
        src = np.squeeze(nussl.AudioSignal(original_src_paths[i]).audio_data.T)[0:signal_len]
        for j in xrange(len(src)):
            reference_sources[i][j] += src[j]

    extracted_sources = np.array(extracted_src_list)

    return mir_eval.separation.bss_eval_sources(reference_sources, extracted_sources)


def hellinger_distance(means, covars):
    """
    calculates the hellinger distance between two distributions described by means and covariances
    :param means:
    :param covars:
    :return:
    """
    mean_diff = means[0] - means[1]
    sigma_bar = (covars[0] + covars[1]) / 2
    exponent = (-1/8) * (mean_diff.T) * np.linalg.inv(sigma_bar) * mean_diff
    term1 = (covars[0]**.25) * (covars[1]**.25) / (sigma_bar **.5)
    term2 = np.exp(exponent)

    return 1-(term1*term2)


def bc_distance(means, covars):
    """
    calculates the bhattacharyya distance between two distributions
    :param means:
    :param covars:
    :return:
    """
    sigma_bar = (np.matrix(covars[0]) + np.matrix(covars[1])) / 2
    mean_diff = (means[0] - means[1]).reshape(1, 2)
    term1 = mean_diff.T * mean_diff
    term1 *= np.linalg.inv(sigma_bar)
    term1 /= 8

    term2 = .5 * np.log((np.linalg.det(sigma_bar))/(np.sqrt(np.linalg.det(covars[0])*np.linalg.det(covars[1]))))

    return term1 + term2


def bc_distance_univariate(means, variances):

    term1 = .25*np.log(.25*((variances[0]**2 / variances[1]**2) + (variances[1]**2 / variances[0]**2) + 2))
    term2 = .25 * ((means[0] - means[1])**2 / ((variances[0]**2) + (variances[1]**2)))
    return term1 + term2


def kl_divergence(means, covars):
    """
    calculates the k-l divergence between two distributions
    :param means:
    :param covars:
    :return:
    """
    mean_diff = (means[0] - means[1]).reshape(2, 1)
    term1 = np.log(np.abs(covars[1])/np.abs(covars[0]))
    term2 = np.trace(np.linalg.inv(covars[1])*covars[0])
    term3 = mean_diff.T * np.linalg.inv(covars[1]) * mean_diff
    K = 3
    return .5 * (term1 + term2 + term3 - K)


def kl_divergence_univariate(means, vars):
    """
    computes the summed kl divergence between two univariate distributions
    :param means:
    :param vars:
    :return:
    """
    term1 = ((means[0] - means[1])**2 + ((vars[0]**2) + (vars[1]**2))) * .5
    term2 = (1/(vars[1]**2)) + (1/(vars[0]**2))
    return (term1 * term2) - 2


def euclidiean_distance(mean0, mean1, var_x, var_y):
    """
    calculates the distance between mean 0 and mean 1 in units of variance
    :param mean0:
    :param mean1:
    :param var_x:
    :param var_y:
    :return:
    """
    diff = mean1-mean0
    dist_x = diff[0] / var_x
    dist_y = diff[1] / var_y

    return np.sqrt(dist_x**2 + dist_y**2)


def run_duet(src_dir, dest_dir, limit=0, plot = False, use_sdr = False, save_sources = False, fname=None, fit_gmm =False, gmm_fname=None, save_to_file = False):
    """
    runs duet on all sources in src_dir and saves the result in dest_dir.
    :param src_dir:
    :param dest_dir:
    :param limit:
    :return:
    """
    count = 0

    for f in os.listdir(src_dir):
        print f
        sr, data = wav.read(src_dir+ f)
        if data.dtype == np.dtype("int16"):
            data = data / float(np.iinfo(data.dtype).max)

        sig = nussl.AudioSignal(audio_data_array=data.T, sample_rate=sr)

        duet = nussl.Duet(sig, 2)
        duet.run()

        if fit_gmm:
            gmm_data = []
            mahal_dists = []
            hist = duet.non_normalized_hist

            for i in xrange(hist.shape[0]):
                for j in xrange(hist.shape[1]):
                    for k in xrange(int(hist[i][j]/10)):
                        gmm_data.append([i, j])
            g = mixture.GMM(n_components=2, covariance_type='full')
            g.fit(gmm_data)

            bc_dist_x = bc_distance_univariate([g.means_[0][0], g.means_[1][0]],
                                               [g.covars_[0][0][0], g.covars_[1][0][0]])
            bc_dist_y = bc_distance_univariate([g.means_[0][1], g.means_[1][1]],
                                               [g.covars_[0][1][1], g.covars_[1][1][1]])

            kl_dist_x = kl_divergence_univariate([g.means_[0][0], g.means_[1][0]],
                                                 [g.covars_[0][0][0], g.covars_[1][0][0]])
            kl_dist_y = kl_divergence_univariate([g.means_[0][1], g.means_[1][1]],
                                                 [g.covars_[0][1][1], g.covars_[1][1][1]])

            eu_dist_1 = euclidiean_distance(g.means_[0], g.means_[1], g.covars_[0][0][0], g.covars_[0][1][1])
            eu_dist_2 = euclidiean_distance(g.means_[1], g.means_[0], g.covars_[1][0][0], g.covars_[1][1][1])

            average_variance1 = np.average(np.diag(g.covars_[0]))
            average_variance2 = np.average(np.diag(g.covars_[1]))

            if save_to_file:
                with open(gmm_fname, 'a') as a:
                    a.write(str(f) + '\t')
                    a.write(str([bc_dist_x, bc_dist_y]) + '\t')
                    a.write(str([kl_dist_y, kl_dist_x]) + '\t')
                    a.write(str([eu_dist_1, eu_dist_2]) + '\t')
                    a.write(str([average_variance1, average_variance2]) +'\t')
                    a.write('\n')

        if use_sdr:
            original_src_list = os.path.splitext(f)[0].split('_')[0].split('+')
            note = os.path.splitext(f)[0].split('_')[1:len(os.path.splitext(f)[0].split('_'))]

            for i in xrange(0, len(original_src_list)):
                original_src_list[i] = 'audio/seed/'+original_src_list[i]+'.wav'
            (sdr, sir, sar, perm) = calculate_sdrs(duet.separated_sources, original_src_list)

            if save_to_file:
                with open(fname, 'a') as a:
                    for i in original_src_list:
                        a.write(i + '\t')
                    a.write(str(sdr)+ '\t')
                    a.write(str(sir) + '\t')
                    a.write(str(sar) + '\t')
                    a.write(str(perm) + '\t')
                    a.write(str(note) + '\t')
                    a.write('\n')

        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        if plot:
            duet.plot(dest_dir + os.path.splitext(f)[0] + '_3d.png', True)
            duet.plot(dest_dir + os.path.splitext(f)[0] + '_unnormalized_3d.png', True, normalize=False)

        if save_sources:
            output_name_stem = dest_dir + os.path.splitext(f)[0] + '_duet_source'
            i = 1
            for s in duet.make_audio_signals():
                output_file_name = output_name_stem + str(i) + '.wav'
                s.write_audio_to_file(output_file_name, sample_rate=sr)
                i += 1
        count += 1
        if count == limit:
            break


def main():
    #generate_dry_mixtures('audio/seed/', 'audio/mix/')
    #generate_attenuation_varied_mixtures('audio/seed/', 'audio/mix_attn/', [3])
    #generate_impulse_responses('audio/IR/')
    #generate_reverb_mixtures('audio/mix/', 'audio/IR/downloaded/', 'audio/reverb_mix/', [1, 2, 5])

    if not os.path.exists('audio/output/'):
        os.mkdir('audio/output/')
    if not os.path.exists('audio/output/pickle/'):
        os.mkdir('audio/output/pickle/')

    #run_duet('audio/reverb_mix/', 'audio/output/reverb/', use_sdr=True, fname='reverb_sdr.txt', save_to_file=True)
    run_duet('audio/reverb_mix/', 'audio/output/reverb/', fit_gmm=True, save_to_file=True, gmm_fname='reverb_gmm.txt')
    plot_from_txt('reverb_sdr.txt', 'reverb_gmm.txt', 'output/plots/')
    #run_duet('audio/reverb_mix/', 'audio/output/reverb/', use_sdr = True, plot=True, fname='reverb_sdr.txt')
    #run_duet('audio/tests/', 'Output/test/', 'Output/pickle/tests/')
    #run_duet('audio/mix_attn/', 'audio/output/mix_attn/', use_sdr=True)

    #calculate_sdrs('Output/pickle/mix/', 'audio/seed/')


def plot_from_txt(sdr_fname, statistic_fname, output_folder):

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(statistic_fname, 'r') as stat_f:
        stats = {}
        q = csv.DictReader(stat_f, delimiter='\t', fieldnames=['filename', 'bc', 'kl', 'eu', 'var'])
        for line in q:
            for entry in line.keys():
                if entry is not None:
                    if entry not in stats.keys():
                        stats[entry] = []
                    if entry == 'filename':
                        stats[entry].append(line[entry])
                    else:
                        stats[entry].append(np.fromstring(line[entry].translate(None,'[]()'), sep=','))

        with open(sdr_fname, 'r') as sdr_f:
            sdr_stats = {}
            q2 = csv.DictReader(sdr_f, delimiter='\t', fieldnames=['filename1', 'filename2', 'sdr', 'sir', 'sar', 'perm', 'irname'])
            for line2 in q2:
                for entry in line2.keys():
                    if entry is not None:
                        if entry not in sdr_stats.keys():
                            sdr_stats[entry] = []
                        if 'name' in entry:
                            sdr_stats[entry].append(line2[entry])
                        else:
                            sdr_stats[entry].append(np.fromstring(line2[entry].translate(None,'][()'), sep=' '))

        plot_stats = dict()

        plot_stats['numiter'] = []
        for i in xrange(len(stats['filename'])):
            plot_stats['numiter'].append(float(os.path.splitext(stats['filename'][i])[0].split('-')[1]))

        # pick different bc values
        plot_stats['avg_bc'] = []
        for i in stats['bc']:
            plot_stats['avg_bc'].append(np.average(i))

        plot_stats['max_bc'] = []
        for i in stats['bc']:
            plot_stats['max_bc'].append(max(i))

        plot_stats['min_bc'] = []
        for i in stats['bc']:
            plot_stats['min_bc'].append(min(i))

        # pick different kl values
        plot_stats['avg_kl'] = []
        for i in stats['kl']:
            plot_stats['avg_kl'].append(np.average(i))

        plot_stats['max_kl'] = []
        for i in stats['kl']:
            plot_stats['max_kl'].append(max(i))

        plot_stats['min_kl'] = []
        for i in stats['kl']:
            plot_stats['min_kl'].append(min(i))

        #pick different eu values
        plot_stats['avg_eu'] = []
        for i in stats['eu']:
            plot_stats['avg_eu'].append(np.average(i))

        plot_stats['max_eu'] = []
        for i in stats['eu']:
            plot_stats['max_eu'].append(max(i))

        plot_stats['min_eu'] = []
        for i in stats['eu']:
            plot_stats['min_eu'].append(min(i))

        # pick different var values
        plot_stats['avg_var'] = []
        for i in stats['var']:
            plot_stats['avg_var'].append(np.average(i))

        plot_stats['max_var'] = []
        for i in stats['var']:
            plot_stats['max_var'].append(max(i))

        plot_stats['min_var'] = []
        for i in stats['var']:
            plot_stats['min_var'].append(min(i))

        plot_stats['sdr'] = []
        for i in sdr_stats['sdr']:
            plot_stats['sdr'].append(np.average(i))

# region Plotting iterations versus stats
        plt.scatter(plot_stats['numiter'], plot_stats['avg_bc'])
        plt.title('avg Bhatacharyya coefficient versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg Bhatacharyya coef')
        plt.ylim(0,4)
        plt.grid(True)
        plt.savefig(output_folder+'1.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['max_bc'])
        plt.title('max Bhatacharyya coefficient versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max Bhatacharyya coef')
        plt.ylim(0,4)
        plt.grid(True)
        plt.savefig(output_folder+'2.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['min_bc'])
        plt.title('min Bhatacharyya coefficient versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min Bhatacharyya coef')
        plt.ylim(0,4)
        plt.grid(True)
        plt.savefig(output_folder+'3.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['avg_kl'])
        plt.title('avg K-L divergence versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg K-L divergence')
        plt.ylim(0,50)
        plt.grid(True)
        plt.savefig(output_folder+'4.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['max_kl'])
        plt.title('max K-L divergence versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max K-L divergence')
        plt.ylim(0,50)
        plt.grid(True)
        plt.savefig(output_folder+'5.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['min_kl'])
        plt.title('min K-L divergence versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min K-L divergence')
        plt.ylim(0,50)
        plt.grid(True)
        plt.savefig(output_folder+'6.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['avg_eu'])
        plt.title('avg Euclidean Distance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg Distance (num stdev)')
        plt.ylim(0,100)
        plt.grid(True)
        plt.savefig(output_folder+'7.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['max_eu'])
        plt.title('max Euclidean Distance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max Distance (num stdev)')
        plt.ylim(0,100)
        plt.grid(True)
        plt.savefig(output_folder+'8.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['min_eu'])
        plt.title('min Euclidean Distance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min Distance (num stdev)')
        plt.ylim(0,100)
        plt.grid(True)
        plt.savefig(output_folder+'9.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['avg_var'])
        plt.title('avg Variance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg Variance')
        plt.grid(True)
        plt.savefig(output_folder+'10.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['max_var'])
        plt.title('max Variance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max Variance')
        plt.grid(True)
        plt.savefig(output_folder+'11.jpg')
        plt.close()

        plt.scatter(plot_stats['numiter'], plot_stats['min_var'])
        plt.title('min Variance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min Variance')
        plt.grid(True)
        plt.savefig(output_folder+'12.jpg')
        plt.close()
#endregion
        pass

#region Plotting stats versus sdrs
        plt.scatter(plot_stats['numiter'], plot_stats['sdr'])
        plt.title('SDR versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.savefig(output_folder+'13.jpg')
        plt.close()

        plt.scatter(plot_stats['avg_bc'], plot_stats['sdr'])
        plt.title('SDR versus avg Bhatacharya Coef')
        plt.xlabel('Bhatacharya Coef')
        plt.ylabel('SDR')
        plt.xlim(0,4)
        plt.grid(True)
        plt.savefig(output_folder+'14.jpg')
        plt.close()

        plt.scatter(plot_stats['max_bc'], plot_stats['sdr'])
        plt.title('SDR versus max Bhatacharya Coef')
        plt.xlabel('Bhatacharya Coef')
        plt.ylabel('SDR')
        plt.xlim(0,4)
        plt.grid(True)
        plt.savefig(output_folder+'15.jpg')
        plt.close()

        plt.scatter(plot_stats['min_bc'], plot_stats['sdr'])
        plt.title('SDR versus min Bhatacharya Coef')
        plt.xlabel('Bhatacharya Coef')
        plt.ylabel('SDR')
        plt.xlim(0, 2)
        plt.grid(True)
        plt.savefig(output_folder+'16.jpg')
        plt.close()

        plt.scatter(plot_stats['avg_kl'], plot_stats['sdr'])
        plt.title('SDR versus avg K-L divergence')
        plt.xlabel('K-L divergence')
        plt.ylabel('SDR')
        plt.xlim(0, 50)
        plt.grid(True)
        plt.savefig(output_folder+'17.jpg')
        plt.close()

        plt.scatter(plot_stats['max_kl'], plot_stats['sdr'])
        plt.title('SDR versus max K-L divergence')
        plt.xlabel('K-L divergence')
        plt.ylabel('SDR')
        plt.xlim(0, 50)
        plt.grid(True)
        plt.savefig(output_folder+'18.jpg')
        plt.close()

        plt.scatter(plot_stats['min_kl'], plot_stats['sdr'])
        plt.title('SDR versus min K-L divergence')
        plt.xlabel('K-L divergence')
        plt.ylabel('SDR')
        plt.xlim(0, 50)
        plt.grid(True)
        plt.savefig(output_folder+'19.jpg')
        plt.close()

        plt.scatter(plot_stats['avg_eu'], plot_stats['sdr'])
        plt.title('SDR versus avg Euclidean Distance')
        plt.xlabel('Distance (num stdev)')
        plt.ylabel('SDR')
        plt.xlim(0, 100)
        plt.grid(True)
        plt.savefig(output_folder+'20.jpg')
        plt.close()

        plt.scatter(plot_stats['max_eu'], plot_stats['sdr'])
        plt.title('SDR versus max Euclidean Distance')
        plt.xlabel('Distance (num stdev)')
        plt.ylabel('SDR')
        plt.xlim(0, 100)
        plt.grid(True)
        plt.savefig(output_folder+'21.jpg')
        plt.close()

        plt.scatter(plot_stats['min_eu'], plot_stats['sdr'])
        plt.title('SDR versus min Euclidean Distance')
        plt.xlabel('Distance (num stdev)')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xlim(0, 5)
        plt.savefig(output_folder+'22.jpg')
        plt.close()

        plt.scatter(plot_stats['avg_var'], plot_stats['sdr'])
        plt.title('SDR versus avg Variance')
        plt.xlabel('Variance')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.savefig(output_folder+'23.jpg')
        plt.close()

        plt.scatter(plot_stats['max_var'], plot_stats['sdr'])
        plt.title('SDR versus max Variance')
        plt.xlabel('Variance')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.savefig(output_folder+'24.jpg')
        plt.close()

        plt.scatter(plot_stats['min_var'], plot_stats['sdr'])
        plt.title('SDR versus min Variance')
        plt.xlabel('Variance')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.savefig(output_folder+'25.jpg')
        plt.close()


#endregion

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


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

    plt.savefig(output_folder+'.jpg')

if __name__ == '__main__':
    main()
