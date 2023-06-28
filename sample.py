# # $^3\rm{He}(\alpha,\alpha)$ Scattering
# 
# Samples capture and Barnard data using `AZURE2` and `emcee`.

import sys
import os
from datetime import datetime
import emcee
import numpy as np
# from multiprocessing import Pool
from schwimmbad import MPIPool

import constants as const

os.environ['OMP_NUM_THREADS'] = '1'

# Define the model.
import model_bayes as model

ndim = model.ndim
nwalkers = 96
nsteps = 10000
nthin = 10

p0 = np.array([[dist.rvs() for dist in model.p0_dist] for i in range(nwalkers)])

pool = MPIPool()

if not pool.is_master():
    pool.wait()
    sys.exit(0)

# Set up the backend.
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
backend_filename = f'samples/{model.__name__}_{dt_string}.h5'
backend = emcee.backends.HDFBackend(backend_filename)
backend.reset(nwalkers, ndim)

print(f'model = {model.__name__} | input file = {model.input_filename}')
print(f'Saving to {backend_filename}.')

moves = [(emcee.moves.DESnookerMove(), 0.8), (emcee.moves.DEMove(), 0.2)]

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    model.ln_posterior,
    moves=moves,
    pool=pool,
    backend=backend
)

for sample in sampler.sample(p0, iterations=nsteps, progress=True, tune=True,
                             thin_by=nthin):
    try:
        tau = sampler.get_autocorr_time()
        if np.all(tau * 100 < sampler.iteration) and np.all(tau != np.zeros(model.azr.config.nd)):
            break
    except:
        pass
