'''
    model_1 with
    * fixed E_x^(7/2-)
'''

import os
import sys
import numpy as np
from scipy import stats
import constants as const

from brick import AZR

input_filename = __name__ + '.azr'
output_filenames = [
    'AZUREOut_aa=1_R=2.out',
    'AZUREOut_aa=1_R=3.out',
    'AZUREOut_aa=1_TOTAL_CAPTURE.out',
    'AZUREOut_aa=1_R=1.out'
]

def filter_by_energy(data, max_energy):
    ii = np.where(data[:, 0] < max_energy)[0]
    return data[ii, :]


max_lab_energy = 3.5 # MeV, cutoff

xs2_data = filter_by_energy(
        np.loadtxt('output/'+output_filenames[0]),
        max_lab_energy
) # capture XS to GS

xs3_data = filter_by_energy(
    np.loadtxt('output/'+output_filenames[1]),
    max_lab_energy
) # capture XS to ES

xs_data = filter_by_energy(
    np.loadtxt('output/'+output_filenames[2]),
    max_lab_energy
) # total capture

xs1_data = filter_by_energy(
    np.loadtxt('output/'+output_filenames[3]),
    max_lab_energy
) # scattering data

scatter = xs1_data[:, 5]
scatter_err = np.sqrt(xs1_data[:, 6]**2 + (0.018*scatter)**2)
xs2 = xs2_data[:, 5]
xs2_err = xs2_data[:, 6]
xs3 = xs3_data[:, 5]
xs3_err = xs3_data[:, 6]

br = xs2
br_err = xs3_err

y = np.hstack((br, xs_data[:, 5], scatter))
dy = np.hstack((br_err, xs_data[:, 6], scatter_err))

azr = AZR(input_filename)
azr.ext_capture_file = 'output/intEC.dat'
azr.root_directory = '/tmp/'
azr.command = '/Applications/AZURE2.app/Contents/MacOS/AZURE2'

rpar_labels = [p.label for p in azr.parameters]

f_labels = [
    '$f_{Seattle}$', '$f_{Weizmann}$', '$f_{LUNA}$', '$f_{ERNA}$', '$f_{ND}$', '$f_{ATOMKI}$',
    '$f_{239}$', '$f_{291}$', '$f_{432}$', '$f_{586}$', '$f_{711}$',
    '$f_{873-1}$', '$f_{873-2}$'
]

def calculate(theta):
    paneru, capture_gs, capture_es, capture_tot = azr.predict(theta)

    bratio = capture_es.xs_com_fit/capture_gs.xs_com_fit
    sigma_tot = capture_tot.xs_com_fit
    scatter_dxs = paneru.xs_com_fit
    return np.hstack((bratio, sigma_tot, scatter_dxs))


# starting position distributions
starting_positions = azr.config.get_input_values().copy()
p0_dist = [stats.norm(sp, np.abs(sp)/100) for sp in starting_positions]

def my_truncnorm(mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


# priors
anc429_prior = stats.uniform(1, 4)
Ga_1hm_prior = stats.uniform(-200e6, 400e6)

Ga_1hp_prior = stats.uniform(0, 100e6)
Gg0_1hp_prior = stats.uniform(0, 10e6)
Gg429_1hp_prior = stats.uniform(-10e3, 20e3)

anc0_prior = stats.uniform(1, 4)
Ga_3hm_prior = stats.uniform(-100e6, 200e6)

Ga_3hp_prior = stats.uniform(0, 100e6)
Gg0_3hp_prior = stats.uniform(-10e3, 20e3)
Gg429_3hp_prior = stats.uniform(-3e3, 6e3)

Ga_5hm_prior = stats.uniform(0, 100e6)
Ga_5hp_prior = stats.uniform(0, 100e6)
Gg0_5hp_prior = stats.uniform(-100e6, 200e6)

Ex_7hm_prior = stats.uniform(1, 9)
Ga_7hm_prior = stats.uniform(0, 10e6)
Gg0_7hm_prior = stats.uniform(0, 1e3)

f_seattle_prior = my_truncnorm(1, 0.03, 0, np.inf)
f_weizmann_prior = my_truncnorm(1, 0.037, 0, np.inf)
f_luna_prior = my_truncnorm(1, 0.032, 0, np.inf)
f_erna_prior = my_truncnorm(1, 0.05, 0, np.inf)
f_nd_prior = my_truncnorm(1, 0.08, 0, np.inf)
f_atomki_prior = my_truncnorm(1, 0.06, 0, np.inf)

# Add 1% (in quadrature) to systematic uncertainty to account for beam-position
# uncertainty.
som_syst = np.sqrt(const.som_syst**2 + 0.01**2)
som_priors = [my_truncnorm(1, syst, 0, np.inf) for syst in som_syst]

priors = [
    anc429_prior,
    Ga_1hm_prior,
    Ga_1hp_prior,
    Gg0_1hp_prior,
    Gg429_1hp_prior,
    anc0_prior,
    Ga_3hm_prior,
    Ga_3hp_prior,
    Gg0_3hp_prior,
    Gg429_3hp_prior,
    Ga_5hm_prior,
    Ga_5hp_prior,
    Gg0_5hp_prior,
    # Ex_7hm_prior,
    Ga_7hm_prior,
    Gg0_7hm_prior,
    f_seattle_prior,
    f_weizmann_prior,
    f_luna_prior,
    f_erna_prior,
    f_nd_prior,
    f_atomki_prior
] + som_priors


def ln_prior(theta):
    return np.sum([prior.logpdf(theta_i) for (prior, theta_i) in zip(priors, theta)])


def ln_likelihood(mu):
    return np.sum(-np.log(const.M_SQRT2PI*dy) - 0.5*((y-mu)/dy)**2)


systematic_uncertainties = np.hstack((const.capture_syst, const.som_syst))

def chi_squared(theta):
    mu = calculate(theta)
    fs = np.array(theta[azr.config.n1:])
    chi2a = np.sum( ((y-mu)/dy)**2 )
    chi2b = np.sum( ((fs - 1)/systematic_uncertainties)**2 )
    return chi2a + chi2b

'''
    log(Posterior)
'''

# Make sure this matches with what log_posterior returns.
blobs_dtype = [('loglikelihood', float)]

def ln_posterior(theta):
    lnpi = ln_prior(theta)
    if lnpi == -np.inf:
        return -np.inf, -np.inf

    mu = calculate(theta)
    lnl = ln_likelihood(mu)
    if np.isnan(lnl):
        return -np.inf, -np.inf

    return lnl + lnpi, lnl
