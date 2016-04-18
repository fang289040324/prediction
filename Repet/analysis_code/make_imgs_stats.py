import Code.nussl
import os
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab
import numpy as np

def main():
    folder = 'training/audio/'
    out = 'training/'

    stats = {}
    # i = 0
    # seen = set()
    # plt.figure().set_size_inches(8,8)

    for file in [f for f in os.listdir(folder) if isfile(join(folder, f))]:
        if splitext(file)[1] == '.wav':
            try:
                a = Code.nussl.AudioSignal(join(folder, file))
                #print('Read {}!'.format(file))
            except:
                print('Couldn\'t read {}'.format(file))
                continue
            #continue

            repet = Code.nussl.Repet(a)

            name_base, n = file.rsplit('_', 1)
            num = n.rsplit('.', 1)[0]

            img_name = splitext(file)[0] + '.png'
            beat_spec = repet.get_beat_spectrum()
            beat_spec_norm = beat_spec / max(beat_spec)

            if num not in stats:
                stats[num] = {'std': [], 'mean': []}


            stats[num]['std'].append(np.std(beat_spec))
            stats[num]['mean'].append(np.mean(beat_spec))

            # if i > 56:
            #     break
            # i += 1

            #print('wrote {}'.format(img_name))

    st = 'std'
    plt.yscale('log')
    plt.xlim(-0.025, 0.25)
    for number in stats.iterkeys():
        plt.plot([float(number)]*len(stats[number][st]), stats[number][st], '.')
    plt.title('Beat spectrum standard deviations')
    plt.xlabel('Randomness amount')
    plt.savefig(join(out, 'beat_spec_std.png'))

    plt.close('all')
    st = 'mean'
    plt.yscale('log')
    plt.xlim(-0.025, 0.25)
    for number in stats.iterkeys():
        plt.plot([float(number)]*len(stats[number][st]), stats[number][st], '.')
    plt.title('Beat spectrum mean')
    plt.xlabel('Randomness amount')
    plt.savefig(join(out, 'beat_spec_mean.png'))









if __name__ == '__main__':
    main()