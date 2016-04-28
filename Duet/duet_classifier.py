import numpy as np
import csv
from sklearn import cross_validation
from sklearn import neighbors
from sklearn import svm
from matplotlib import pyplot as plt
import copy
import itertools


def run_knn(x, y, n):
    neigh = neighbors.KNeighborsRegressor(n_neighbors=n, weights='distance')
    scores = cross_validation.cross_val_score(neigh, x, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predictions = cross_validation.cross_val_predict(neigh, x, y, cv=10)
    return predictions


def run_svm(x, y):
    s = svm.SVR()
    scores = cross_validation.cross_val_score(s, x, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predictions = cross_validation.cross_val_predict(s, x, y, cv=10)

    return predictions


def main():
    stats = make_dicts('reverb_pan_full_sdr.txt', 'reverb_pan_full_stats.txt', 'reverb_pan_full_gmm.txt')
    keys = ['norm_means', 'avg_smoothed', 'diff_smoothed', 'min_var']

    perform_experiment(10, stats, keys)
    # x = np.array([np.array(stats['entropy'][i]) for i in xrange(len(stats['sdr']))])
    # for k in keys:
    #     x = np.array([np.append(x[j], np.log(stats[k][j] + 0.0001)) for j in xrange(len(stats['sdr']))])
    # predictions = run_knn(x, stats['sdr'],10)
    # diff = predictions - np.array(stats['sdr'])
    # print diff.mean(), diff.std()
    # print max(stats['sdr']), min(stats['sdr'])
    # print max(np.abs(diff))
    # plt.hist(diff, bins=20)
    # plt.show()


def perform_experiment(n_folds, stats, keys):
    all_diffs = []

    kf = cross_validation.KFold(len(stats['sdr']), n_folds=n_folds)

    x = np.array([np.array(stats['entropy'][i]) for i in xrange(len(stats['sdr']))])
    for k in keys:
        x = np.array([np.append(x[j], np.log(stats[k][j] + 0.0001)) for j in xrange(len(stats['sdr']))])

    y = np.array(stats['sdr'])
    run = 1
    for train, test in kf:
        print 'Doing run ', run

        x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
        knn = neighbors.KNeighborsRegressor(n_neighbors=10, weights='distance')
        knn.fit(x_train, y_train)

        diffs = []
        for i in xrange(len(x_test)):

            guess = knn.predict(x_test[i].reshape(1,-1))
            diffs.append(y_test[i] - guess)

        all_diffs.append(np.array(diffs))
        run += 1

    # these two lines make it look pretty
    plt.style.use('bmh')
    plt.hist(all_diffs, histtype='stepfilled', stacked=True, alpha=0.8, bins=30)

    plt.title('Generated data histogram')
    plt.xlabel('True SDR $-$ Predicted SDR (dB)')
    plt.show()

    # print out statistics about each of the runs
    mean, std1, std2 = [], [], []
    i = 1
    for diff_list in all_diffs:
        std = np.std(diff_list)
        mean.append(np.mean(diff_list))
        std1.append(std)
        per = float(sum([1 for n in diff_list if np.abs(n) >= 2 * std])) / float(len(diff_list)) * 100
        std2.append(per)
        print 'Run ', str(i)
        print 'Mean = {0:.2f} dB'.format(np.mean(diff_list)), ' Std. Dev. = {0:.2f} dB'.format(std),
        print ' Min = {0:.2f} dB'.format(np.min(diff_list)), ' Max = {0:.2f} dB'.format(np.max(diff_list)),
        print ' ==== % more than 2 std = {0:.2f}%'.format(per)
        i += 1

    print '=' * 80
    print 'Avg. Mean = {0:.2f} dB'.format(np.mean(mean)), 'Avg. Std. Dev = {0:.2f} dB'.format(np.mean(std1)),
    print 'Avg. % more than 2 std = {0:.2f}%'.format(np.mean(std2))

    print 'max =', np.amax(np.array(all_diffs), 1), ' min =', np.min(np.array(all_diffs), 1)


def do_all_n(stats, n):
    keys = stats.keys()
    keys.remove('numiter')
    keys.remove('sdr')
    keys.remove('bc_y')

    for key in itertools.combinations(keys, n):

        print key
        x = np.array([np.array(stats['entropy'][i]) for i in xrange(len(stats['sdr']))])
        for k in key:
            x = np.array([np.append(x[j], np.log(stats[k][j] + 0.0001)) for j in xrange(len(stats['sdr']))])
        predictions = run_knn(x, stats['sdr'], 10)
        diff = predictions - np.array(stats['sdr'])
        print diff.mean(), diff.std()


def do_all_pairs(stats):
    visited = ['sdr', 'numiter']
    for i in stats.keys():
        if i in visited:
            continue
        visited.append(i)
        X1 = np.array([np.array([stats[i][k]]) for k in xrange(len(stats['sdr']))])
        print i
        predictions = run_knn(X1, stats['sdr'], 5)
        diff = predictions - np.array(stats['sdr'])
        print diff.mean(), diff.std()

        if not i == 'entropy':

            pad = 0.0001

            print 'entropy ', i
            X = np.array([np.append(stats[i][k], stats['entropy'][k]) for k in xrange(len(stats['sdr']))])
            print 'knn:'
            predictions = run_knn(X, stats['sdr'], 5)
            diff = predictions - np.array(stats['sdr'])
            print diff.mean(), diff.std()
            print 'svm:'
            predictions = run_svm(X, stats['sdr'])
            diff = predictions - np.array(stats['sdr'])
            print diff.mean(), diff.std()

            print 'entropy ', 'log', i
            X = np.array([np.append(np.log(stats[i][k]+pad), stats['entropy'][k]) for k in xrange(len(stats['sdr']))])
            print 'knn:'
            predictions = run_knn(X, stats['sdr'], 5)
            diff = predictions - np.array(stats['sdr'])
            print diff.mean(), diff.std()
            print 'svm:'
            predictions = run_svm(X, stats['sdr'])
            diff = predictions - np.array(stats['sdr'])
            print diff.mean(), diff.std()


def make_dicts(sdr_fname, stat_fname, gmm_fname):
    with open(gmm_fname, 'r') as stat_f:
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
                        stats[entry].append(np.fromstring(line[entry].translate(None, '[]()'), sep=','))

    with open(stat_fname, 'r') as stat2_f:
        stats2 = {}
        q3 = csv.DictReader(stat2_f, delimiter='\t', fieldnames=['filename', 'peaks', 'smoothed_peaks', 'entropy', 'means', 'norm_means', 'smooth_means'])
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
        q2 = csv.DictReader(sdr_f, delimiter='\t',
                            fieldnames=['filename1', 'filename2', 'sdr', 'sir', 'sar', 'perm', 'note'])
        for line2 in q2:
            for entry in line2.keys():
                if entry is not None:
                    if entry not in sdr_stats.keys():
                        sdr_stats[entry] = []
                    if 'name' in entry or 'note' in entry:
                        sdr_stats[entry].append(line2[entry])
                    else:
                        sdr_stats[entry].append(np.fromstring(line2[entry].translate(None, '][()').replace("\'", ""), sep=' '))

    plot_stats = dict()

    plot_stats['numiter'] = []
    plot_stats['avg_bc'] = []
    plot_stats['max_bc'] = []
    plot_stats['min_bc'] = []
    plot_stats['bc_x'] = []
    plot_stats['bc_y'] = []
    plot_stats['avg_kl'] = []
    plot_stats['max_kl'] = []
    plot_stats['min_kl'] = []
    plot_stats['avg_eu'] = []
    plot_stats['max_eu'] = []
    plot_stats['min_eu'] = []
    plot_stats['avg_var'] = []
    plot_stats['max_var'] = []
    plot_stats['min_var'] = []
    plot_stats['sdr'] = []
    plot_stats['avg_peak'] = []
    plot_stats['min_peak'] = []
    plot_stats['max_peak'] = []
    plot_stats['diff_peak'] = []
    plot_stats['avg_smoothed'] = []
    plot_stats['min_smoothed'] = []
    plot_stats['max_smoothed'] = []
    plot_stats['diff_smoothed'] = []
    plot_stats['entropy'] = []
    plot_stats['means'] = []
    plot_stats['norm_means'] = []
    plot_stats['smooth_means'] = []

    # pick different bc values
    for i in xrange(len(stats['bc'])):
        if np.average(sdr_stats['sdr'][i]) < -100:
            continue
        if '-5' in sdr_stats['note'][i] or '-4' in sdr_stats['note'][i]:
            continue
        plot_stats['avg_bc'].append(np.average(stats['bc'][i]))
        plot_stats['max_bc'].append(max(stats['bc'][i]))
        plot_stats['min_bc'].append(min(stats['bc'][i]))
        plot_stats['bc_x'].append(stats['bc'][i][0])
        plot_stats['bc_y'].append(stats['bc'][i][1])
        plot_stats['avg_kl'].append(np.average(stats['kl'][i]))
        plot_stats['max_kl'].append(max(stats['kl'][i]))
        plot_stats['min_kl'].append(min(stats['kl'][i]))
        plot_stats['avg_eu'].append(np.average(stats['eu'][i]))
        plot_stats['max_eu'].append(max(stats['eu'][i]))
        plot_stats['min_eu'].append(min(stats['eu'][i]))
        plot_stats['avg_var'].append(np.average(stats['var'][i]))
        plot_stats['max_var'].append(max(stats['var'][i]))
        plot_stats['min_var'].append(min(stats['var'][i]))
        plot_stats['sdr'].append(np.average(sdr_stats['sdr'][i]))
        plot_stats['avg_peak'].append(np.average(stats2['peaks'][i]))
        plot_stats['max_peak'].append(max(stats2['peaks'][i]))
        plot_stats['min_peak'].append(min(stats2['peaks'][i]))
        plot_stats['diff_peak'].append(abs(stats2['peaks'][i][0] - stats2['peaks'][i][1]))
        plot_stats['avg_smoothed'].append(np.average(stats2['smoothed_peaks'][i]))
        plot_stats['max_smoothed'].append(max(stats2['smoothed_peaks'][i]))
        plot_stats['min_smoothed'].append(min(stats2['smoothed_peaks'][i]))
        plot_stats['diff_smoothed'].append(
            abs(stats2['smoothed_peaks'][i][0] - stats2['smoothed_peaks'][i][1]))
        plot_stats['entropy'].append(stats2['entropy'][i][0])
        plot_stats['means'].append(stats2['means'][i][0])
        plot_stats['norm_means'].append(stats2['norm_means'][i][0])
        plot_stats['smooth_means'].append(stats2['smooth_means'][i][0])

    return plot_stats

if __name__ == '__main__':
    main()
