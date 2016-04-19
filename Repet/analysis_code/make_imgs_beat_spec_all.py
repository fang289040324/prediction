import nussl
import os
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def main():
    folder = 'training/audio/'
    out = 'training/beat_spec_all'

    seen = set()
    first = True
    name_base = 'hum1_blah1'
    plt.figure().set_size_inches(8,8)

    for file in [f for f in os.listdir(folder) if isfile(join(folder, f))]:
        if splitext(file)[1] == '.wav':
            try:
                a = nussl.AudioSignal(join(folder, file))
                #print('Read {}!'.format(file))
            except:
                print('Couldn\'t read {}'.format(file))
                continue
            #continue

            repet = nussl.Repet(a)

            prev_name_base = name_base
            name_base, n = file.rsplit('_', 1)
            n = n.rsplit('.', 1)[0]

            img_name = splitext(file)[0] + '.png'
            beat_spec = repet.get_beat_spectrum()

            beat_spec_norm = beat_spec / max(beat_spec)


            if name_base not in seen:
                if not first:
                    ax = plt.subplot(111)
                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.title(prev_name_base)
                    plt.savefig(join(out, prev_name_base + '.png'))

                plt.close('all')
                seen.add(name_base)
                first = False
                #return

            #plt.grid('on')

            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.8
            left_h = left + width + 0.02
            rect_bs = [left, bottom, width, height]
            rect_y = [left_h, bottom, 0.2, height]

            #bs = plt.axes(rect_bs)
            #y_hist = plt.axes(rect_y)

            #bs.plot(beat_spec)
            plt.plot(beat_spec, label=n)
            #y_hist.hist(beat_spec, bins=30, orientation='horizontal', facecolor='blue', histtype='stepfilled')
            nullfmt = NullFormatter()
            #y_hist.yaxis.set_major_formatter(nullfmt)
            #y_hist.set_xscale('log')

            #plt.title('Beat Spectrum for {}'.format(file))










if __name__ == '__main__':
    main()