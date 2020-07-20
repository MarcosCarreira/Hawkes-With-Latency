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
support = 2


# %% [markdown]
# #### Using a function

# %% Function g1
def g1(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * t)


# %% Function g2 (latency)
def g2(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * (t - 1))\
        * np.heaviside(t -1, 1)


# %% Define xgrid
xgrid = np.linspace(0, support, 20+1)

# %% Define tf1
tf1 = TimeFunction((xgrid, g1(xgrid)), inter_mode=TimeFunction.InterConstRight)

# %% Define tf2
tf2 = TimeFunction((xgrid, g2(xgrid)), inter_mode=TimeFunction.InterConstRight)

# %% Plot tf1
plot_timefunction(tf1)

# %% Plot tf2
plot_timefunction(tf2)

# %% [markdown]
# ## Defining Non-parametric Kernels

# %% [markdown]
# ## Defining Parametric Kernels

# %% [markdown]
# ## Simulations

# %% [markdown]
# ## Learning

# %%
