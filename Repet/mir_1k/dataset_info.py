import nussl, os

if __name__ == '__main__':
    mir_1k_folder = 'MIR-1K/UndividedWavfile/'
    files = [os.path.join(mir_1k_folder, f) for f in os.listdir(mir_1k_folder) if '.wav' in f]

    lengths = []
    for file in files:
        s = nussl.AudioSignal(file)
        lengths.append(s.signal_duration)

    print 'min = ', min(lengths), 'max = ', max(lengths)
