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

# %%
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

# %%
from tick.hawkes import SimuHawkes, SimuHawkesMulti

# %% [markdown]
# #### Non-parametric

# %%
from tick.base import TimeFunction
from tick.hawkes import HawkesKernelTimeFunc

# %% [markdown]
# #### Parametric

# %%
from tick.hawkes import HawkesKernelSumExp

# %% [markdown]
# ### Learners

# %% [markdown]
# #### Non-parametric

# %%
from tick.hawkes import HawkesEM, HawkesBasisKernels

# %% [markdown]
# #### Parametric

# %%
from tick.hawkes import HawkesSumExpKern, HawkesSumGaussians

# %% [markdown]
# ### Plotting kernels

# %%
from tick.plot import plot_timefunction
from tick.plot import plot_point_process
from tick.plot import plot_hawkes_kernel_norms, plot_hawkes_kernels
from tick.plot import plot_basis_kernels

# %% [markdown]
# ## Defining Time Functions

# %%
support = 20


# %% [markdown]
# #### Using a function

# %%
def g1(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * t)


# %%
def g2(t):
    return 0.7 * 0.5 * 10 * np.exp(-0.5 * 10 * (t - 1) * np.heaviside(t -1, 0))


# %%
xgrid = np.linspace(0, support, 20+1)

# %%
xgrid

# %%
tf1 = TimeFunction((xgrid, g1(xgrid)))

# %%
tf2 = TimeFunction((xgrid, g2(xgrid)))

# %%
plot_timefunction(tf1)

# %%
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
