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

# data files to which we will apply a normalization factor
# (not really, we apply it to the theory calculations)
capture_data_files = [
    'Seattle_XS.dat',
    'Weizmann_XS.dat',
    'LUNA_XS.dat',
    'ERNA_XS.dat',
    'ND_XS.dat',
    'ATOMKI_XS.dat',
]

scatter_data_files = [
    'sonik_inflated_239.dat',
    'sonik_inflated_291.dat',
    'sonik_inflated_432.dat',
    'sonik_inflated_586.dat',
    'sonik_inflated_711.dat',
    'sonik_inflated_873_1.dat',
    'sonik_inflated_873_2.dat'
]

use_brune = True
observable_only = False

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

x = np.hstack((xs2_data[:, 0], xs_data[:, 0], xs1_data[:, 0]))
# y = np.hstack((br, xs_data[:, 5], scatter))
# dy = np.hstack((br_err, xs_data[:, 6], scatter_err))

nbr = xs2_data.shape[0]
nxs = xs_data.shape[0]
nscatter1 = xs1_data.shape[0]

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

labels = rpar_labels + f_labels

nrpar = len(rpar_labels)
nf_capture = len(capture_data_files)
nf_scatter = len(scatter_data_files)
ndim = nrpar + nf_capture + nf_scatter

assert ndim == len(labels), f'''
Number of sampled parameters does not match the number of labels: {ndim} ≠ {len(labels)}.
'''

def map_uncertainty(theta, ns):
    c = np.array([])
    for (theta_i, n) in zip(theta, ns):
        c = np.hstack((c, theta_i*np.ones(n)))
    return c


def calculate(theta):
    sonik, capture_gs, capture_es, capture_tot = azr.predict(theta)

    # data
    bratio_data = capture_es.xs_com_data/capture_gs.xs_com_data
    y = np.hstack((bratio_data, capture_tot.xs_com_data, sonik.xs_com_data))
    dy = np.hstack((br_err, capture_tot.xs_err_com_data, sonik.xs_err_com_data))


    bratio = capture_es.xs_com_fit/capture_gs.xs_com_fit
    return np.hstack((bratio, capture_tot.xs_com_fit, sonik.xs_com_fit)), y, dy


starting_positions = azr.config.get_input_values().copy()

assert ndim == np.size(starting_positions), f'''
Number of sampled parameters ({ndim}) does not match the number of starting
parameters ({starting_positions.size}).
'''

# starting position distributions
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

assert len(priors) == ndim, f'''
Number of priors ({len(priors)}) does not match number of sampled parameters ({ndim}).
'''

def ln_prior(theta):
    return np.sum([prior.logpdf(theta_i) for (prior, theta_i) in zip(priors, theta)])


def ln_likelihood(mu, y, dy):
    return np.sum(-np.log(const.M_SQRT2PI*dy) - 0.5*((y-mu)/dy)**2)


'''
    log(Posterior)
'''

# Make sure this matches with what log_posterior returns.
blobs_dtype = [('loglikelihood', float)]

def ln_posterior(theta):
    lnpi = ln_prior(theta)
    if lnpi == -np.inf:
        return -np.inf, -np.inf

    mu, y, dy = calculate(theta)
    lnl = ln_likelihood(mu, y, dy)
    if np.isnan(lnl):
        return -np.inf, -np.inf

    return lnl + lnpi, lnl
