import numpy as np


def bc_distance_univariate(means, variances):

    """
    Calculates the univariate Bhattacharyya_distance between two distributions. See https://en.wikipedia.org/wiki/Bhattacharyya_distance
    Args:
        means: an array of means
        variances: an array of variances.

    Returns:
        the Bhattacharyya distance between the distributions
    """
    term1 = .25*np.log(.25*((variances[0]**2 / variances[1]**2) + (variances[1]**2 / variances[0]**2) + 2))
    term2 = .25 * ((means[0] - means[1])**2 / ((variances[0]**2) + (variances[1]**2)))
    return term1 + term2


def kl_divergence_univariate(means, vars):
    """
    computes the summed kl divergence between two univariate distributions
    :param means: an array of means of the distributions
    :param vars: an array of variances of the distributions
    :return: the k-l divergence between the distributions
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