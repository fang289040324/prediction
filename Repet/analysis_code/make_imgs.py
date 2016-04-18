import nussl
import os
from os.path import join, isfile, splitext
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab
import numpy as np

def main():
    folder = 'training/audio/'
    out = 'training/beat_spec'

    # seen = set()
    # plt.figure().set_size_inches(8,8)

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

            name_base, n = file.rsplit('_', 1)
            n = n.rsplit('.', 1)[0]

            img_name = splitext(file)[0] + '.png'
            beat_spec = repet.get_beat_spectrum()

            beat_spec_norm = beat_spec / max(beat_spec)


            #plt.grid('on')

            left, width = 0.1, 0.65
            bottom, height = 0.1, 0.8
            left_h = left + width + 0.02
            rect_bs = [left, bottom, width, height]
            rect_y = [left_h, bottom, 0.2, height]

            plt.close('all')
            bs = plt.axes(rect_bs)
            y_hist = plt.axes(rect_y)

            bs.plot(beat_spec)
            #bs.title('Beat Spectrum for {}'.format(file))
            num, bins, patches = y_hist.hist(beat_spec, bins=30, orientation='horizontal')
            #line = mlab.normpdf(bins, np.mean(beat_spec), np.std(beat_spec))
            #print np.mean(beat_spec), np.std(beat_spec)
            #y_hist.plot(bins, line)

            nullfmt = NullFormatter()
            y_hist.yaxis.set_major_formatter(nullfmt)
            y_hist.set_xscale('log')



            plt.savefig(join(out, img_name))
            print('wrote {}'.format(img_name))

            return









if __name__ == '__main__':
    main()