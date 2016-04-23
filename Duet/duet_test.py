import nussl
import mir_eval

if __name__ == '__main__':
    sig = nussl.AudioSignal()

    duet = nussl.Duet(sig, 2)
    duet()

