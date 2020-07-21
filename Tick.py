# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tick library - guide and testing

# %% [markdown]
# ## Python Imports

# %% Python imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlwings as xw
import itertools

# %% [markdown]
# ## Tick Imports

# %% [markdown]
# ### Simulation

# %% Import Simulation
from tick.hawkes import SimuHawkes, SimuHawkesMulti

# %% [markdown]
# #### Non-parametric

# %% Import Time Function
from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc

# %% [markdown]
# #### Parametric

# %% Import HawkesKernelSumExp
from tick.hawkes import HawkesKernelSumExp

# %% [markdown]
# ### Learners

# %% [markdown]
# #### Non-parametric

# %% Import Non-parametric Learners
from tick.hawkes import HawkesEM, HawkesBasisKernels

# %% [markdown]
# #### Parametric

# %% Import Parametric Learners
from tick.hawkes import HawkesSumExpKern, HawkesSumGaussians

# %% [markdown]
# ### Plotting kernels

# %% Import Plots
from tick.plot import plot_timefunction
from tick.plot import plot_point_process
from tick.plot import plot_hawkes_kernel_norms, plot_hawkes_kernels
from tick.plot import plot_basis_kernels

# %% [markdown]
# ## Defining Time Functions

# %% Define Support
support = 4


# %% [markdown]
# #### Using a function

# %% Function g1
def g1(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * t)


# %% [markdown]
# Introducing a latency with heaviside

# %% Function g2 (latency)
def g2(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * (t - 1))\
        * np.heaviside(t -1, 1) # To ensure zero before t0=1


# %% Define xgrid
xgrid = np.linspace(0, support, 40+1)

# %% Define tf1
tf1 = TimeFunction((xgrid, g1(xgrid)), inter_mode=TimeFunction.InterLinear)

# %% Define tf2
tf2 = TimeFunction((xgrid, g2(xgrid)), inter_mode=TimeFunction.InterConstRight)

# %% Plot tf1
plot_timefunction(tf1)

# %% Plot tf2
plot_timefunction(tf2)


# %% [markdown]
# ### Creating a complete function with latency and linear interpolation

# %% Define the time_function
def time_func(f, support, t0=0, steps=1000):
    t_values = np.linspace(0, support, steps+1)
    y_values = f(t_values - t0) * np.heaviside(t_values - t0, 1)
    return TimeFunction(values=(t_values, y_values),
                        border_type=TimeFunction.Border0,
                        inter_mode=TimeFunction.InterLinear)


# %% Define tf_1
tf_1 = time_func(g1, 4)

# %% Define tf_2
tf_2 = time_func(g1, 4, 1)

# %% Check equivalent values
print([[tf_1.value(0), tf_1.value(1)], [tf_2.value(1), tf_2.value(2)]])

# %% Plot tf_1
plot_timefunction(tf_1)

# %% Plot tf_2
plot_timefunction(tf_2)

# %% [markdown]
# ## Defining Non-parametric Kernels

# %% Define kernel_1
kernel_1 = HawkesKernelTimeFunc(tf_1)

# %% Define kernel_1
kernel_2 = HawkesKernelTimeFunc(tf_2)

# %% [markdown]
# ## Defining Parametric Kernels

# %% Define kernel_sumexp
kernel_sumexp = HawkesKernelSumExp(
    intensities=np.array([0.1, 0.2, 0.1]),
    decays=np.array([1.0, 3.0, 7.0]))

# %% [markdown]
# ## Simulations

# %% [markdown]
# ### SimuHawkes

# %% Define simulation for one realization, sumexp kernel
hawkes = SimuHawkes(n_nodes=1, end_time=40, seed=1398)
hawkes.set_kernel(0, 0, kernel_sumexp)
hawkes.set_baseline(0, 1.)

# %% Track intensity and simulation
dt = 0.01
hawkes.track_intensity(dt)
hawkes.simulate()

# %% Get attributes of hawkes
timestamps = hawkes.timestamps
intensity = hawkes.tracked_intensity
intensity_times = hawkes.intensity_tracked_times
mean_intensity = hawkes.mean_intensity()

# %% Plot jumps
pd.Series(np.arange(1, len(timestamps[0])+1),
          index=timestamps[0]).plot(drawstyle='steps-post')

# %% Plot point express
plot_point_process(hawkes)

# %% [markdown]
# ### SimuHawkesMulti

# %% Use either SumExp kernel or latency
hawkes_m = SimuHawkes(n_nodes=1, end_time=1000)
hawkes_m.set_baseline(0, 1.)

# hawkes_m.set_kernel(0, 0, kernel_sumexp)

# hawkes_m.set_kernel(0, 0, kernel_2)

hawkes_m.set_kernel(0, 0, HawkesKernelTimeFunc(time_func(g1, 4, 0.1)))

# %% Run Multi
multi = SimuHawkesMulti(hawkes_m, n_simulations=10)
multi.simulate()

# %% Get attributes from Multi
multi_timestamps = multi.timestamps
multi_mean_intensity = multi.mean_intensity

# %% [markdown]
# ## Learning

# %% [markdown]
# #### Non-parametric

# %% [markdown]
# ##### HawkesEM

# %% HawkesEM

kern_d = np.linspace(0, 4, 100+1)

em = HawkesEM(kernel_discretization=kern_d, max_iter=1000, tol=1e-5,
              verbose=True)
em.fit(multi_timestamps)
em_baseline = em.baseline
em_kernel = em.kernel
em_score = em.score()

# %% Plot HawkesEM
plot_hawkes_kernels(em, hawkes=hawkes_m)

# %% [markdown]
# ##### HawkesBasisKernels

# %% HawkesBasisKernels
bk = HawkesBasisKernels(kernel_support=4, n_basis=2, kernel_size=100,
                        max_iter=10000, tol=1e-5, C=0.1, verbose=True)
bk.fit(multi_timestamps)
bk_baseline = bk.baseline
bk_amplitudes = bk.amplitudes
bk_kernel = bk.basis_kernels

# %% Plot HawkesBasisKernels
plot_hawkes_kernels(bk, hawkes=hawkes_m)

# %% [markdown]
# #### Parametric

# %% [markdown]
# ##### HawkesSumExpKern

# %% HawkesSumExpKern
solvers = ['agd', 'bfgs']
penalties = ['l2', 'elasticnet']
sek = HawkesSumExpKern(decays=[0.5],
                       n_baselines=1, penalty='elasticnet', solver='agd',
                       elastic_net_ratio=0.8,
                       max_iter=10000, tol=1e-5, C=1000., verbose=True)
sek.fit(multi_timestamps)
sek_baseline = sek.baseline
sek_adjacency = sek.adjacency
sek_score = sek.score()

# %% Plot HawkesSumExpKern
plot_hawkes_kernels(sek, hawkes=hawkes_m)

# %% [markdown]
# ##### HawkesSumGaussians

# %% HawkesSumGaussians
sg = HawkesSumGaussians(max_mean_gaussian=4, n_gaussians=7,
                        lasso_grouplasso_ratio=0.5,
                        max_iter=10000, tol=1e-5, C=1000., verbose=True)
sg.fit(multi_timestamps)
sg_baseline = sg.baseline
sg_amplitudes = sg.amplitudes
sg_means = sg.means_gaussians
sg_std = sg.std_gaussian


# %% Plot HawkesSumGaussians
plot_hawkes_kernels(sg, hawkes=hawkes_m)


# %%
