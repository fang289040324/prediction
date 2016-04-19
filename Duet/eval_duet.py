import os
import nussl
import mir_eval
from sklearn import mixture
import csv
import scipy.io.wavfile as wav
import numpy as np
import distance_measures as d
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


def main():
    # run_duet('audio/reverb_mix/', 'audio/output/reverb/', plot=True, save_sources=True, use_sdr=True, sdr_fname='reverb_sdr.txt',
    #          fit_gmm=True, gmm_fname='reverb_gmm.txt', use_other_stats=True, stats_fname='other_stats.txt')
    plot_from_txt('reverb_sdr.txt', 'reverb_gmm.txt', 'other_stats.txt', 'output/reverb_plots/', per_file=False)
    plot_from_txt('reverb_sdr.txt', 'reverb_gmm.txt', 'other_stats.txt', 'output/reverb_plots_per_file/', per_file=True)


def run_duet(src_dir, dest_dir, plot=False, use_sdr=False, save_sources=False, sdr_fname=None, fit_gmm=False,
             gmm_fname=None, hist_ratio =.1, use_other_stats=False, stats_fname=None):
    """

    Args:
        src_dir:
        dest_dir:
        plot:
        use_sdr:
        save_sources:
        sdr_fname:
        fit_gmm:
        gmm_fname:
        hist_ratio:
        use_other_stats:
        stats_fname:

    Returns:

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
            if gmm_fname is None:
                raise ValueError('Must Pass a gmm statistic filename if you want to save gmm results')

            gmm_data = []
            hist = duet.smoothed_hist

            for i in xrange(hist.shape[0]):
                for j in xrange(hist.shape[1]):
                    for k in xrange(int(hist[i][j]*hist_ratio)):
                        gmm_data.append([i, j])
            g = mixture.GMM(n_components=2, covariance_type='full')
            g.fit(gmm_data)

            print g.means_

            bc_dist_x = d.bc_distance_univariate([g.means_[0][0], g.means_[1][0]],
                                               [g.covars_[0][0][0], g.covars_[1][0][0]])
            bc_dist_y = d.bc_distance_univariate([g.means_[0][1], g.means_[1][1]],
                                               [g.covars_[0][1][1], g.covars_[1][1][1]])

            kl_dist_x = d.kl_divergence_univariate([g.means_[0][0], g.means_[1][0]],
                                                 [g.covars_[0][0][0], g.covars_[1][0][0]])
            kl_dist_y = d.kl_divergence_univariate([g.means_[0][1], g.means_[1][1]],
                                                 [g.covars_[0][1][1], g.covars_[1][1][1]])

            eu_dist_1 = d.euclidiean_distance(g.means_[0], g.means_[1], g.covars_[0][0][0], g.covars_[0][1][1])
            eu_dist_2 = d.euclidiean_distance(g.means_[1], g.means_[0], g.covars_[1][0][0], g.covars_[1][1][1])

            average_variance1 = np.average(np.diag(g.covars_[0]))
            average_variance2 = np.average(np.diag(g.covars_[1]))

            with open(gmm_fname, 'a') as a:
                a.write(str(f) + '\t')
                a.write(str([bc_dist_x, bc_dist_y]) + '\t')
                a.write(str([kl_dist_y, kl_dist_x]) + '\t')
                a.write(str([eu_dist_1, eu_dist_2]) + '\t')
                a.write(str([average_variance1, average_variance2]) +'\t')
                a.write('\n')

        if use_sdr:
            if sdr_fname is None:
                raise ValueError('Must Pass a sdr filename if you want to save sdr results')

            original_src_list = os.path.splitext(f)[0].split('_')[0].split('+')
            note = os.path.splitext(f)[0].split('_')[1:len(os.path.splitext(f)[0].split('_'))]

            for i in xrange(0, len(original_src_list)):
                original_src_list[i] = 'audio/seed/'+original_src_list[i]+'.wav'
            (sdr, sir, sar, perm) = calculate_sdrs(duet.separated_sources, original_src_list)

            with open(sdr_fname, 'a') as a:
                for i in original_src_list:
                    a.write(i + '\t')
                a.write(str(sdr) + '\t')
                a.write(str(sir) + '\t')
                a.write(str(sar) + '\t')
                a.write(str(perm) + '\t')
                a.write(str(note) + '\t')
                a.write('\n')

        if use_other_stats:
            if stats_fname is None:
                raise ValueError('Must Pass a stats filename if you want to save other statistics')

            peaks = [duet.non_normalized_hist[duet.peak_indices[0, 0], duet.peak_indices[1, 0]],
                     duet.non_normalized_hist[duet.peak_indices[0, 1], duet.peak_indices[1, 1]]]
            smoothed_peaks = [duet.smoothed_hist[duet.peak_indices[0, 0], duet.peak_indices[1, 0]],
                              duet.smoothed_hist[duet.peak_indices[0, 1], duet.peak_indices[1, 1]]]

            print [[duet.peak_indices[0, 0], duet.peak_indices[1, 0]], [duet.peak_indices[0, 1], duet.peak_indices[1, 1]]]

            entropy = 0
            hist_sum = np.sum(duet.non_normalized_hist)
            for i in np.nditer(duet.non_normalized_hist):
                prob = (i/hist_sum)
                if prob == 0:
                    continue
                else:
                    entropy += prob * math.log(prob, 2)
            entropy *= -1

            with open(stats_fname, 'a') as a:
                a.write(str(f) + '\t')
                a.write(str(peaks) + '\t')
                a.write(str(smoothed_peaks) + '\t')
                a.write(str(entropy) + '\t')
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


def calculate_sdrs(extracted_src_list, original_src_paths):

    signal_len = len(extracted_src_list[0])
    num_srcs = len(extracted_src_list)

    reference_sources = np.zeros((num_srcs, signal_len))
    for i in xrange(0, len(original_src_paths)):
        src = np.squeeze(nussl.AudioSignal(original_src_paths[i]).audio_data.T)[0:signal_len]
        for j in xrange(len(src)):
            reference_sources[i][j] += src[j]

    extracted_sources = np.array(extracted_src_list)

    if np.any(extracted_sources.max(1) == 0):
        # Can't pass a silent source to mir_eval
        print "silent source"
        return np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), (0, 0)

    return mir_eval.separation.bss_eval_sources(reference_sources, extracted_sources)


def plot_from_txt(sdr_fname, statistic_fname, statistic2_fname, output_folder, per_file=True):

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

    with open(statistic2_fname, 'r') as stat2_f:
        stats2 = {}
        q3 = csv.DictReader(stat2_f, delimiter='\t', fieldnames=['filename', 'peaks', 'smoothed_peaks', 'entropy'])
        for line in q3:
            for entry in line.keys():
                if entry is not None:
                    if entry not in stats2.keys():
                        stats2[entry] = []
                    if entry == 'filename':
                        stats2[entry].append(line[entry])
                    else:
                        stats2[entry].append(np.fromstring(line[entry].translate(None, '[]()'), sep=','))

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

        plot_stats['numiter'] = {}
        plot_stats['avg_bc'] = {}
        plot_stats['max_bc'] = {}
        plot_stats['min_bc'] = {}
        plot_stats['bc_x'] = {}
        plot_stats['bc_y'] = {}
        plot_stats['avg_kl'] = {}
        plot_stats['max_kl'] = {}
        plot_stats['min_kl'] = {}
        plot_stats['avg_eu'] = {}
        plot_stats['max_eu'] = {}
        plot_stats['min_eu'] = {}
        plot_stats['avg_var'] = {}
        plot_stats['max_var'] = {}
        plot_stats['min_var'] = {}
        plot_stats['sdr'] = {}
        plot_stats['avg_peak'] = {}
        plot_stats['min_peak'] = {}
        plot_stats['max_peak'] = {}
        plot_stats['avg_smoothed'] = {}
        plot_stats['min_smoothed'] = {}
        plot_stats['max_smoothed'] = {}
        plot_stats['entropy'] = {}

        for i in xrange(len(stats['filename'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            for j in plot_stats.keys():
                if mix_name not in plot_stats[j].keys():
                    plot_stats[j][mix_name] = []

        for i in xrange(len(stats['filename'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            plot_stats['numiter'][mix_name].append(float(os.path.splitext(stats['filename'][i])[0].split('-')[1]))

        # pick different bc values

        for i in xrange(len(stats['bc'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            plot_stats['avg_bc'][mix_name].append(np.average(stats['bc'][i]))
            plot_stats['max_bc'][mix_name].append(max(stats['bc'][i]))
            plot_stats['min_bc'][mix_name].append(min(stats['bc'][i]))
            plot_stats['bc_x'][mix_name].append(stats['bc'][i][0])
            plot_stats['bc_y'][mix_name].append(stats['bc'][i][1])

        # pick different kl values
        for i in xrange(len(stats['kl'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            plot_stats['avg_kl'][mix_name].append(np.average(stats['kl'][i]))
            plot_stats['max_kl'][mix_name].append(max(stats['kl'][i]))
            plot_stats['min_kl'][mix_name].append(min(stats['kl'][i]))

        #pick different eu values
        for i in xrange(len(stats['eu'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            plot_stats['avg_eu'][mix_name].append(np.average(stats['eu'][i]))
            plot_stats['max_eu'][mix_name].append(max(stats['eu'][i]))
            plot_stats['min_eu'][mix_name].append(min(stats['eu'][i]))

        # pick different var values
        for i in xrange(len(stats['var'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            plot_stats['avg_var'][mix_name].append(np.average(stats['var'][i]))
            plot_stats['max_var'][mix_name].append(max(stats['var'][i]))
            plot_stats['min_var'][mix_name].append(min(stats['var'][i]))

        for i in xrange(len(sdr_stats['sdr'])):
            mix_name = os.path.splitext(stats['filename'][i])[0].split('-')[0]
            plot_stats['sdr'][mix_name].append(np.average(sdr_stats['sdr'][i]))

        for i in xrange(len(stats2['peaks'])):
            mix_name = os.path.splitext(stats2['filename'][i])[0].split('-')[0]
            plot_stats['avg_peak'][mix_name].append(np.average(stats2['smoothed_peaks'][i]))
            plot_stats['max_peak'][mix_name].append(max(stats2['smoothed_peaks'][i]))
            plot_stats['min_peak'][mix_name].append(min(stats2['smoothed_peaks'][i]))
            plot_stats['avg_smoothed'][mix_name].append(np.average(stats2['smoothed_peaks'][i]))
            plot_stats['max_smoothed'][mix_name].append(max(stats2['smoothed_peaks'][i]))
            plot_stats['min_smoothed'][mix_name].append(min(stats2['smoothed_peaks'][i]))
            plot_stats['entropy'][mix_name].append(stats2['entropy'][i])


# region Plotting iterations versus stats
        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['avg_bc'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['avg_bc'][i])
        plt.title('avg Bhatacharyya coefficient versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg Bhatacharyya coef')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'1.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['max_bc'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['max_bc'][i])
        plt.title('max Bhatacharyya coefficient versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max Bhatacharyya coef')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'2.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['min_bc'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['min_bc'][i])
        plt.title('min Bhatacharyya coefficient versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min Bhatacharyya coef')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'3.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['avg_kl'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['avg_kl'][i])
        plt.title('avg K-L divergence versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg K-L divergence')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'4.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['max_kl'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['max_kl'][i])
        plt.title('max K-L divergence versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max K-L divergence')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'5.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['min_kl'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['min_kl'][i])
        plt.title('min K-L divergence versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min K-L divergence')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'6.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['avg_eu'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['avg_eu'][i])
        plt.title('avg Euclidean Distance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg Distance (num stdev)')
        plt.grid(True)
        plt.savefig(output_folder+'7.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['max_eu'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['max_eu'][i])
        plt.title('max Euclidean Distance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max Distance (num stdev)')
        plt.grid(True)
        plt.savefig(output_folder+'8.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['min_eu'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['min_eu'][i])
        plt.title('min Euclidean Distance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min Distance (num stdev)')
        plt.grid(True)
        plt.savefig(output_folder+'9.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['avg_var'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['avg_var'][i])
        plt.title('avg Variance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('avg Variance')
        plt.grid(True)
        plt.savefig(output_folder+'10.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['max_var'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['max_var'][i])
        plt.title('max Variance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('max Variance')
        plt.grid(True)
        plt.savefig(output_folder+'11.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['min_var'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['min_var'][i])
        plt.title('min Variance versus Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min Variance')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder+'12.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['min_peak'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['min_peak'][i])
        plt.title('min Peak Height vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min peak Height')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(output_folder + '29.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['max_peak'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['max_peak'][i])
        plt.title('max Peak Height vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min peak Height')
        plt.grid(True)
        plt.savefig(output_folder + '30.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['avg_peak'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['avg_peak'][i])
        plt.title('avg Peak Height vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min peak Height')
        plt.grid(True)
        plt.savefig(output_folder + '31.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['min_smoothed'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['min_smoothed'][i])
        plt.title('min smoothed Peak Height vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min peak Height')
        plt.grid(True)
        plt.savefig(output_folder + '32.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['max_smoothed'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['max_smoothed'][i])
        plt.title('max smoothed Peak Height vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min peak Height')
        plt.grid(True)
        plt.savefig(output_folder + '33.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['avg_smoothed'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['avg_smoothed'][i])
        plt.title('avg smoothed Peak Height vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('min peak Height')
        plt.grid(True)
        plt.savefig(output_folder + '34.jpg')
        plt.close()

        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['entropy'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['entropy'][i])
        plt.title('entropy vs Number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('entropy')
        plt.grid(True)
        plt.savefig(output_folder + '35.jpg')
        plt.close()
#endregion

#region Plotting stats versus sdrs
        for i in plot_stats['numiter'].keys():
            if per_file:
                plt.plot(plot_stats['numiter'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['numiter'][i], plot_stats['sdr'][i])
        plt.title('SDR versus number of iterations')
        plt.xlabel('Number of iterations')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.savefig(output_folder+'13.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['avg_bc'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['avg_bc'][i], plot_stats['sdr'][i])
        plt.title('SDR versus avg Bhatacharya Coef')
        plt.xlabel('Bhatacharya Coef')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'14.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['max_bc'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['max_bc'][i], plot_stats['sdr'][i])
        plt.title('SDR versus max Bhatacharya Coef')
        plt.xlabel('Bhatacharya Coef')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'15.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['min_bc'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['min_bc'][i], plot_stats['sdr'][i])
        plt.title('SDR versus min Bhatacharya Coef')
        plt.xlabel('Bhatacharya Coef')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'16.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['avg_kl'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['avg_kl'][i], plot_stats['sdr'][i])
        plt.title('SDR versus avg K-L divergence')
        plt.xlabel('K-L divergence')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'17.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['max_kl'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['max_kl'][i], plot_stats['sdr'][i])
        plt.title('SDR versus max K-L divergence')
        plt.xlabel('K-L divergence')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'18.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['min_kl'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['min_kl'][i], plot_stats['sdr'][i])
        plt.title('SDR versus min K-L divergence')
        plt.xlabel('K-L divergence')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'19.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['avg_eu'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['avg_eu'][i], plot_stats['sdr'][i])
        plt.title('SDR versus avg Euclidean Distance')
        plt.xlabel('Distance (num stdev)')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'20.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['max_eu'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['max_eu'][i], plot_stats['sdr'][i])
        plt.title('SDR versus max Euclidean Distance')
        plt.xlabel('Distance (num stdev)')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'21.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['min_eu'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['min_eu'][i], plot_stats['sdr'][i])
        plt.title('SDR versus min Euclidean Distance')
        plt.xlabel('Distance (num stdev)')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'22.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['avg_var'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['avg_var'][i], plot_stats['sdr'][i])
        plt.title('SDR versus avg Variance')
        plt.xlabel('Variance')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'23.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['max_var'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['max_var'][i], plot_stats['sdr'][i])
        plt.title('SDR versus max Variance')
        plt.xlabel('Variance')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'24.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['min_var'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['min_var'][i], plot_stats['sdr'][i])
        plt.title('SDR versus min Variance')
        plt.xlabel('Variance')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'25.jpg')
        plt.close()

        for i in stats['bc']:
            plt.scatter(i[0], i[1])
        plt.title('bcdelay vs bcattn')
        plt.xlabel('bcdelay')
        plt.ylabel('bcattn')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder+'26.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['bc_x'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['bc_x'][i], plot_stats['sdr'][i])
        plt.title('SDR versus Bhattacharya coef delay')
        plt.xlabel('BC_delay')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '27.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['bc_y'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['bc_y'][i], plot_stats['sdr'][i])
        plt.title('SDR versus Bhattacharya coef attn')
        plt.xlabel('BC_attn')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '28.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['min_peak'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['min_peak'][i], plot_stats['sdr'][i])
        plt.title('SDR versus min Peak Height')
        plt.xlabel('Peak Height')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '36.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['max_peak'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['max_peak'][i], plot_stats['sdr'][i])
        plt.title('SDR versus max Peak Height')
        plt.xlabel('Peak Height')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '37.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['avg_peak'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['avg_peak'][i], plot_stats['sdr'][i])
        plt.title('SDR versus avg Peak Height')
        plt.xlabel('Peak Height')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '38.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['min_smoothed'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['min_smoothed'][i], plot_stats['sdr'][i])
        plt.title('SDR versus min smoothed Peak Height')
        plt.xlabel('Peak Height')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '39.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['max_smoothed'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['max_smoothed'][i], plot_stats['sdr'][i])
        plt.title('SDR versus max smoothed Peak Height')
        plt.xlabel('Peak Height')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '40.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['avg_smoothed'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['avg_smoothed'][i], plot_stats['sdr'][i])
        plt.title('SDR versus avg smoothed Peak Height')
        plt.xlabel('Peak Height')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(output_folder + '41.jpg')
        plt.close()

        for i in plot_stats['sdr'].keys():
            if per_file:
                plt.plot(plot_stats['entropy'][i], plot_stats['sdr'][i])
            plt.scatter(plot_stats['entropy'][i], plot_stats['sdr'][i])
        plt.title('SDR versus entropy')
        plt.xlabel('Entropy')
        plt.ylabel('SDR')
        plt.grid(True)
        plt.savefig(output_folder + '42.jpg')
        plt.close()


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


if __name__ == '__main__':
    main()
