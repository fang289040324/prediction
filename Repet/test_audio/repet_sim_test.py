# coding=utf-8
"""
Repet sim test
"""

import os
import nussl


def main():
    """
    main
    :return:
    """
    # input audio file
    input_name = os.path.join('HistoryRepeating_PropellorHeads.wav')
    signal = nussl.AudioSignal(path_to_input_file=input_name)

    # make a directory to store output if needed
    if not os.path.exists('../Output/'):
        os.mkdir('../Output')

    # L = 4096
    # signal.stft_params.window_length = L
    # signal.stft_params.window_type = nussl.WindowType.HANN

    # Set up Repet
    repet = nussl.Repet(signal, repet_type=nussl.RepetType.SIM)
    # and Run
    # repet.plot('prelude_electric_piano_bs.png')
    repet.run()

    # Get foreground and backgroun audio signals
    bkgd, fgnd = repet.make_audio_signals()

    # and write out to files
    bkgd.write_audio_to_file(os.path.join('..', 'Output', 'repet_sim_background.wav'))
    fgnd.write_audio_to_file(os.path.join('..', 'Output', 'repet_sim_foreground.wav'))


if __name__ == '__main__':
    main()