import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def log_density(coefs, y, x1):

    [alpha, beta1, 
     sigma2u, sigma2v] = coefs[:]

    Lambda = np.sqrt(sigma2u / sigma2v)
    sigma2 = sigma2u + sigma2v
    sigma = np.sqrt(sigma2)

    # Composed errors from the production function equation
    eps = y - alpha - x1 * beta1 

    # Compute the log density
    Den = (
        (2 / sigma)
        * stats.norm.pdf(eps / sigma)
        * stats.norm.cdf(-Lambda * eps / sigma)
    )
    logDen = np.log(Den)

    return logDen





def loglikelihood(coefs, y, x1):

    logDen = log_density(coefs, y, x1)
    log_likelihood = -np.sum(logDen)

    return log_likelihood

def estimate(y, x1, b_true, w_true, noise_std_u, noise_std_v):

    # Starting values for MLE
    alpha = np.log(float(b_true))
    beta1 = float(w_true)
    sigma2u = noise_std_u**2
    sigma2v = noise_std_v**2

    theta0 = np.array([alpha, beta1,
                       sigma2u, sigma2v])

    bounds = [(None, None) for x in range(len(theta0) - 2)] + [
        (1e-6, np.inf),
        (1e-6, np.inf),
    ]

    # Minimize the negative log-likelihood using numerical optimization.
    MLE_results = minimize(
        fun=loglikelihood,
        x0=theta0,
        method="L-BFGS-B",
        tol = 1e-6,
        options={"ftol": 1e-6, "maxiter": 1000, "maxfun": 6*1000},
        args=(y, x1),
        bounds=bounds,
    )

    theta = MLE_results.x  
    log_likelihood = MLE_results.fun  

    # Estimate standard errors
    delta = 1e-6
    grad = np.zeros((len(y), len(theta)))
    for i in range(len(theta)):
        theta1 = np.copy(theta)
        theta1[i] += delta
        grad[:, i] = (
            log_density(theta1, y, x1) - log_density(theta, y, x1)
        ) / delta

    OPG = grad.T @ grad  
    ster = np.sqrt(np.diag(np.linalg.inv(OPG)))

    return theta, ster, log_likelihood