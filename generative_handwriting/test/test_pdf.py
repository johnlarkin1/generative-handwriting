import numpy as np


def calculate_pdf(x, y, mu1, mu2, sigma1, sigma2, rho):
    """
    Calculate the value of a bivariate Gaussian PDF.
    """

    z = (
        np.square((x - mu1) / sigma1)
        + np.square((y - mu2) / sigma2)
        - 2 * rho * (x - mu1) * (y - mu2) / (sigma1 * sigma2)
    )
    denom = 2 * np.pi * sigma1 * sigma2 * np.sqrt(1 - np.square(rho))
    result = np.exp(-z / (2 * (1 - np.square(rho)))) / denom
    return result
