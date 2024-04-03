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


pis = np.array([0.20205513 0.19969349 0.19926226 0.19986431 0.1991248 ]
sigma1 = [1.0441458 1.0378072 1.034876  1.0140684 1.0506142]
sigma2 = [1.1515504 1.2389483 1.1596389 1.136923  1.1977159]
rhos = [-0.0116948  -0.01468305 -0.01948133  0.00585194  0.00388459]
mu1 = [0.01924138 0.00081639 0.01265204 0.0105293  0.00950682]
mu2 = [-0.00759727 -0.01743073 -0.02131062 -0.02066256 -0.01559294]
eos 0.49622905
