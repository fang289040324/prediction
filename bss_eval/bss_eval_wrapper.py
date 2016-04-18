import matlab_wrapper
import os


def bss_eval_sources_wrapper(estimated_signal, real_signal, cur_matlab_root=None):
    """
    Wrapper to call into bss_eval_sources in matlab. This will make it easier to call bss_eval from python.
    :param estimated_signal: numpy array with dimensions [length, ch]
    :param real_signal: numpy array with dimensions [length, ch]
    :param cur_matlab_root: location of the matlab executable. To find this out, type matlabroot in matlab
    and copy that path into here. If this function crashes without it, you probably need to add that string.
    :return: list with [sdr, sar, sir, perm] values from bss_eval_sources
    """
    if cur_matlab_root is None:
        matlab_session = matlab_wrapper.MatlabSession()
    else:
        matlab_session = matlab_wrapper.MatlabSession(matlab_root=cur_matlab_root)

    matlab_session.put('estimated', estimated_signal)
    matlab_session.put('real', real_signal)
    matlab_session.eval('run_bss_eval')

    bss_output = [matlab_session.get('sdr'), matlab_session.get('sar'), matlab_session.get('sir'),
                      matlab_session.get('perm')]

    return bss_output
