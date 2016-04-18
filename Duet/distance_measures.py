import numpy as np


def hellinger_distance(means, covars):
    """
    calculates the hellinger distance between two distributions described by means and covariances
    :param means:
    :param covars:
    :return:
    """
    mean_diff = means[0] - means[1]
    sigma_bar = (covars[0] + covars[1]) / 2
    exponent = (-1/8) * (mean_diff.T) * np.linalg.inv(sigma_bar) * mean_diff
    term1 = (covars[0]**.25) * (covars[1]**.25) / (sigma_bar **.5)
    term2 = np.exp(exponent)

    return 1-(term1*term2)


def bc_distance(means, covars):
    """
    calculates the bhattacharyya distance between two distributions
    :param means:
    :param covars:
    :return:
    """
    sigma_bar = (np.matrix(covars[0]) + np.matrix(covars[1])) / 2
    mean_diff = (means[0] - means[1]).reshape(1, 2)
    term1 = mean_diff.T * mean_diff
    term1 *= np.linalg.inv(sigma_bar)
    term1 /= 8

    term2 = .5 * np.log((np.linalg.det(sigma_bar))/(np.sqrt(np.linalg.det(covars[0])*np.linalg.det(covars[1]))))

    return term1 + term2


def bc_distance_univariate(means, variances):

    term1 = .25*np.log(.25*((variances[0]**2 / variances[1]**2) + (variances[1]**2 / variances[0]**2) + 2))
    term2 = .25 * ((means[0] - means[1])**2 / ((variances[0]**2) + (variances[1]**2)))
    return term1 + term2


def kl_divergence(means, covars):
    """
    calculates the k-l divergence between two distributions
    :param means:
    :param covars:
    :return:
    """
    mean_diff = (means[0] - means[1]).reshape(2, 1)
    term1 = np.log(np.abs(covars[1])/np.abs(covars[0]))
    term2 = np.trace(np.linalg.inv(covars[1])*covars[0])
    term3 = mean_diff.T * np.linalg.inv(covars[1]) * mean_diff
    K = 3
    return .5 * (term1 + term2 + term3 - K)


def kl_divergence_univariate(means, vars):
    """
    computes the summed kl divergence between two univariate distributions
    :param means:
    :param vars:
    :return:
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