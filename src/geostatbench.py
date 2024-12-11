# Import all functions later
__all__ = ["MAE", "RMSE", "AESD", "PSRF", "KS_distance"]

import argparse
import logging
import os
import pathlib
import sys
import glob
import threading
import traceback
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import warnings
from numpy import loadtxt
from easyDataverse import  Dataset
from tqdm import tqdm

import inspect # Get current file path 
import yaml

import matplotlib as mpl
# Configure the backend to use pgf
# mpl.use('svg')
import matplotlib

mpl.rcParams['text.usetex'] = False

    
__author__ = "NilsWildt"
__copyright__ = "NilsWildt"
__license__ = "MIT"
__version__ = "0.1.0"

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
_logger = logging.getLogger(__name__)

# https://github.com/BIDS/colormap
# CC0
# NOTE:
# This is a copy of "parula" from MATLAB. We don't claim any rights to this
# file, but The Mathworks does. Consult them and/or a lawyer if you want to
# use it.

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# https://stackoverflow.com/a/60601041
def add_colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    return cbar


cm_data = [[0.2081, 0.1663, 0.5292], 
           [0.2116238095, 0.1897809524, 0.5776761905],
           [0.212252381, 0.2137714286, 0.6269714286],
           [0.2081, 0.2386, 0.6770857143],
           [0.1959047619, 0.2644571429, 0.7279],
           [0.1707285714, 0.2919380952, 0.779247619],
           [0.1252714286, 0.3242428571, 0.8302714286],
           [0.0591333333, 0.3598333333, 0.8683333333],
           [0.0116952381, 0.3875095238, 0.8819571429],
           [0.0059571429, 0.4086142857, 0.8828428571],
           [0.0165142857, 0.4266, 0.8786333333],
           [0.032852381, 0.4430428571, 0.8719571429],
           [0.0498142857, 0.4585714286, 0.8640571429],
           [0.0629333333, 0.4736904762, 0.8554380952],
           [0.0722666667, 0.4886666667, 0.8467],
           [0.0779428571, 0.5039857143, 0.8383714286],
           [0.079347619, 0.5200238095, 0.8311809524],
           [0.0749428571, 0.5375428571, 0.8262714286],
           [0.0640571429, 0.5569857143, 0.8239571429],
           [0.0487714286, 0.5772238095, 0.8228285714],
           [0.0343428571, 0.5965809524, 0.819852381], 
           [0.0265, 0.6137, 0.8135],
           [0.0238904762, 0.6286619048, 0.8037619048],
           [0.0230904762, 0.6417857143, 0.7912666667],
           [0.0227714286, 0.6534857143, 0.7767571429],
           [0.0266619048, 0.6641952381, 0.7607190476],
           [0.0383714286, 0.6742714286, 0.743552381],
           [0.0589714286, 0.6837571429, 0.7253857143],
           [0.0843, 0.6928333333, 0.7061666667],
           [0.1132952381, 0.7015, 0.6858571429],
           [0.1452714286, 0.7097571429, 0.6646285714],
           [0.1801333333, 0.7176571429, 0.6424333333],
           [0.2178285714, 0.7250428571, 0.6192619048],
           [0.2586428571, 0.7317142857, 0.5954285714],
           [0.3021714286, 0.7376047619, 0.5711857143],
           [0.3481666667, 0.7424333333, 0.5472666667],
           [0.3952571429, 0.7459, 0.5244428571],
           [0.4420095238, 0.7480809524, 0.5033142857],
           [0.4871238095, 0.7490619048, 0.4839761905],
           [0.5300285714, 0.7491142857, 0.4661142857],
           [0.5708571429, 0.7485190476, 0.4493904762],
           [0.609852381, 0.7473142857, 0.4336857143], 
           [0.6473, 0.7456, 0.4188],
           [0.6834190476, 0.7434761905, 0.4044333333],
           [0.7184095238, 0.7411333333, 0.3904761905],
           [0.7524857143, 0.7384, 0.3768142857],
           [0.7858428571, 0.7355666667, 0.3632714286],
           [0.8185047619, 0.7327333333, 0.3497904762],
           [0.8506571429, 0.7299, 0.3360285714],
           [0.8824333333, 0.7274333333, 0.3217],
           [0.9139333333, 0.7257857143, 0.3062761905],
           [0.9449571429, 0.7261142857, 0.2886428571],
           [0.9738952381, 0.7313952381, 0.266647619],
           [0.9937714286, 0.7454571429, 0.240347619],
           [0.9990428571, 0.7653142857, 0.2164142857],
           [0.9955333333, 0.7860571429, 0.196652381],
           [0.988, 0.8066, 0.1793666667],
           [0.9788571429, 0.8271428571, 0.1633142857],
           [0.9697, 0.8481380952, 0.147452381],
           [0.9625857143, 0.8705142857, 0.1309],
           [0.9588714286, 0.8949, 0.1132428571],
           [0.9598238095, 0.9218333333, 0.0948380952],
           [0.9661, 0.9514428571, 0.0755333333], 
           [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)

#https://plotly.com/python/v3/matplotlib-colorscales/
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
            
plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False  # don't setup fonts from rc parameters
})
mpl.rcParams['agg.path.chunksize'] = 10000

def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def plot_mcmc_trace(X):
    xint = range(1, len(X) + 1)
    fig, ax = plt.subplots(figsize=set_size(300), dpi=150)
    p1 = sns.scatterplot(ax=ax,
                         x=xint,
                         y=X,
                         s=2,
                         alpha=0.05,
                         color=".2",
                         marker=".",
                         edgecolor="black")
    ax.set_xlabel("Sample count")
    # ax.set_ylabel("$\lambda 1$")
    return p1


def plot_mcmc_trace_hist(X):
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(np.arange(X.shape[0]), X)
    plt.title('Trace Plot for $\\lambda 1$')

    plt.subplot(122)
    plt.hist(X, orientation='horizontal', bins=80)
    plt.title('Histogram for $\\lambda 1$')

    plt.tight_layout()
    plt.show()


def plot_mcmc_histogram(X):
    fig, ax = plt.subplots(figsize=set_size(300), dpi=150)
    p1 = sns.histplot(ax=ax, x=X, kde=True)
    ax.set_xlabel(r"$\lambda 1$")
    # ax.set_ylabel("$\lambda 1$")
    return p1


def plot_mcmc_ecdf(X):
    fig, ax = plt.subplots(figsize=set_size(300), dpi=150)
    p1 = sns.histplot(ax=ax,
                      x=X,
                      element="step",
                      fill=False,
                      cumulative=True,
                      stat="frequency",
                      common_norm=False)
    return p1



def lollipop_plot(categories, values, ylim=None):
    # Number of categories
    num_categories = len(categories)
    # Ensure values are in the range [0, 1] for proper plotting
    values = np.array(values)
    values = (values - np.min(values)) / (np.max(values) - np.min(values))
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot lollipops
    ax.vlines(categories, ymin=0, ymax=values, color='b', linewidth=2, alpha=0.5)
    ax.plot(categories, values, 'o', markersize=8, color='b', alpha=0.5)

    # Set xtick labels (categories)
    ax.set_xticks(categories)
    ax.set_xticklabels(categories, fontsize=12, rotation=45)  # Adjust rotation as needed

    # Customize y-axis and ylim
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Remove the y-axis labels (value numbers)
    ax.set_yticklabels([])

    # Add grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig, ax





def radar_plot(categories, values, ylim=None):
    """
    radar plot

    
    parameters
    ----------
    categories: list
        labels around the circle
    values: ndarray
        values for each category
    ylim: list
        y (r) value limits 

    return
    ------
    fig, ax
    
    """

    # repeat the first value to close the circular graph
    values = np.append(values, values[0]) 

    # calculate  label locations (angle values)
    theta = np.linspace(start=0, stop=2 * np.pi, num=len(values))
    
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    # set zero location to the north
    ax.set_theta_zero_location('N')

    # set xtick labels (categories)
    locs = ax.set_xticks(theta[:-1])
    labels = ax.set_xticklabels(categories, fontsize=11) #, color='grey')

    # avoid some overlaps
    half = (len(labels) - 1) // 2
    for t in labels[1:half]:
        t.set_horizontalalignment('right')
    for t in labels[-half + 1:]:
        t.set_horizontalalignment('left')

    # set r label positions
    ax.set_rlabel_position(-60)

    # plot 
    ax.plot(theta, values)
    ax.fill(theta, values, 'b', alpha=0.1)
    if ylim is not None:
        ax.set_ylim(ylim)
        
    ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), fontsize=11) #, color="grey")
    # ax.set_yticklabels(ax.get_yticklabels(), size=10) #, color="grey")
    # ax.tick_params('y', labelsize=11)

    return fig, ax



def MAE(y_ref: np.ndarray, y_can: np.ndarray) -> float:
    """
    Mean absolute error between the posterior statistics of reference and the candidate
    
    Parameters
    ----------
    y_ref : np.ndarray
        posterior statistics of the reference: e.g., posterior mean or standard deviation field
    y_can : np.ndarray
        posterior statistics of the candidate: e.g., posterior mean or standard deviation field

    Returns
    -------
    float
        mean absolute error.

    """
    return np.abs(y_ref - y_can).mean()

def RMSE(y_ref: np.ndarray, y_can: np.ndarray) -> float:
    """
    Root mean square error between the posterior statistics of reference and candidate
    
    Parameters
    ----------
    y_ref : np.ndarray
        posterior statistics of the reference: e.g., posterior mean or standard deviation field
    y_can : np.ndarray
        posterior statistics of the candidate: e.g., posterior mean or standard deviation field

    Returns
    -------
    float
        root mean square error.

    """
    return np.sqrt(np.mean((y_ref - y_can)**2))


def AESD(posterior_std: np.ndarray) -> float:
    """
    Average ensemble standard deviation of posterior
    
    Parameters
    ----------
    posterior_std : np.ndarray
        standard deviation of posterior.

    Returns
    -------
    float
        average ensemble standard deviation.

    """
    return np.mean(posterior_std)


def PSRF(chains) -> float:
    """
    Potential scale reduction factor calculated with multiple chains from MCMC
    
    Parameters
    ----------
    chains: list 
        each element in 'chains' represents one chain.
        shape of each chain: N x () - first dimension denotes the number of samples.
    
    Returns
    -------
    R: float
        potential scale reduction factor.

    """

    # Number of chains
    m = len(chains)
    
    # Lenght of each chain
    n = chains[0].shape[0]

    # First, calculate the wihtin-chain variance W
    W = 0.0
    for j in range(m):
        W = W + chains[j].var(axis=0)
    W = W/m

    # calculate the between-chain variance B
    B = 0.0
    mean_each_chain = np.zeros((m,) + chains[0].shape[1:]) # mean of each chain
    for j in range(m):
        mean_each_chain[j] = chains[j].mean(axis=0)
    mean_all_chains = mean_each_chain.mean(axis=0)

    for j in range(m):
        B = B + (mean_each_chain[j] - mean_all_chains)**2
    B = B * n/(m-1)

    # variance estimator
    estimated_var = (n-1)/n * W + B/n

    # estimated potential scale reduction factor
    R = (estimated_var/W)**0.5
    return R
    

def KS_distance(y_ref: np.ndarray, y_can: np.ndarray) -> float:
    """
    Kolmogorov–Smirnov distance between pointwise posterior distributions of reference and candidate
    
    Parameters
    ----------
    y_ref : np.ndarray
        posterior samples of the reference
    y_can : np.ndarray
        posterior samples of the candidate

    Returns
    -------
    d: float
        KS distance

    """
    y_ref = np.sort(y_ref)
    y_can = np.sort(y_can)
    n1 = y_ref.shape[0]
    n2 = y_can.shape[0]
    data_all = np.concatenate([y_ref, y_can])
    cdf1 = np.searchsorted(y_ref, data_all, side='right') / n1
    cdf2 = np.searchsorted(y_can, data_all, side='right') / n2
    d = np.max(np.absolute(cdf1 - cdf2))
    #     assert d==1.0
    return d

def nonlinear_transform(x):
    """
    nonlinear transform for metrics larger than 1

    Parameters
    ----------
    x: float
        original metric 

    Returns
    -------
    x_t: float
        transformed metric in [0,1]
        
    """
    
    x_t = x / (1 + x)
    return x_t

def compare_distributions_props(Y_ref: np.ndarray, Y_ben: np.ndarray, Y_prior: np.ndarray):
    """
    Compare the properties of distributions between two datasets (Y_ref and Y_ben) and a prior dataset (Y_prior).

    Parameters:
        Y_ref (np.ndarray): The reference dataset.
        Y_ben (np.ndarray): The dataset to be compared.
        Y_prior (np.ndarray): The prior dataset for normalization.

    Returns:
        Data Frame: including all the metrics values.
    """
    _logger.info("Start calculating metrics")
    ## Root Mean Squared Error (RMSE) of posterior mean
    # Calculate the Root Mean Squared Error (RMSE) for the mean of the datasets.
    # This measures the difference between the means of the reference and compared datasets.
    rmse_mean = RMSE(np.mean(Y_ref, axis=0).ravel(), np.mean(Y_ben, axis=0).ravel())
    _logger.info(f"RMSE mean: {rmse_mean}")

    # Calculate the RMSE between the mean of the reference dataset and the mean of the prior dataset.
    rmse_mean_prior = RMSE(np.mean(Y_ref, axis=0).ravel(), np.mean(Y_prior, axis=0).ravel())
    _logger.info(f"RMSE mean prior: {rmse_mean_prior}")

    # Normalize RMSE by dividing the calculated RMSE of the datasets by the RMSE with the prior dataset.
    # This helps to compare how different the means of Y_ref and Y_ben are compared to Y_ref and Y_prior.
    rmse_mean_n = rmse_mean / rmse_mean_prior
    _logger.info(f"Normalized RMSE mean: {rmse_mean_n}")

    # Non-linearly transform rmse_mean_n. (Implementation of the nonlinear_transform function is not shown here.)
    rmse_mean_n_t = nonlinear_transform(rmse_mean_n)
    _logger.info(f"Transformed Normalized RMSE mean: {rmse_mean_n_t}")

    ## Root Mean Squared Error (RMSE) of posterior standard deviation
    
    # Calculate the Root Mean Squared Error (RMSE) for the standard deviation of the datasets.
    # This measures the difference between the standard deviations of the reference and compared datasets.
    rmse_std = RMSE(np.std(Y_ref, axis=0).ravel(), np.std(Y_ben, axis=0).ravel())
    _logger.info(f"RMSE std: {rmse_std}")
    
    # Calculate the RMSE between the standard deviation of the reference dataset and the standard deviation of the prior dataset.
    rmse_std_prior = RMSE(np.std(Y_ref, axis=0).ravel(), np.std(Y_prior, axis=0).ravel())
    _logger.info(f"RMSE std prior: {rmse_std_prior}")
   
    # Normalize RMSE by dividing the calculated RMSE of the datasets by the RMSE with the prior dataset.
    # This helps to compare how different the standard deviations of Y_ref and Y_ben are compared to Y_ref and Y_prior.
    rmse_std_n = rmse_std / rmse_std_prior
    _logger.info(f"Normalized RMSE std: {rmse_std_n}")

    # Non-linearly transform rmse_std_n.
    rmse_std_n_t = nonlinear_transform(rmse_std_n)
    _logger.info(f"Transformed Normalized RMSE std: {rmse_std_n_t}")

    ## Mean Absolute Error (MAE) of posterior mean 
    
    # Calculate the Mean Absolute Error (MAE) for the mean of the datasets.
    # This measures the absolute difference between the means of the reference and compared datasets.
    mae_mean = MAE(np.mean(Y_ref, axis=0).ravel(), np.mean(Y_ben, axis=0).ravel())
    _logger.info(f"MAE mean: {mae_mean}")
    
    # Calculate the MAE between the mean of the reference dataset and the mean of the prior dataset.
    mae_mean_prior = MAE(np.mean(Y_ref, axis=0).ravel(), np.mean(Y_prior, axis=0).ravel())
    _logger.info(f"MAE mean prior: {mae_mean_prior}")
    
    # Normalize MAE by dividing the calculated MAE of the datasets by the MAE with the prior dataset.
    # This helps to compare how different the means of Y_ref and Y_ben are compared to Y_ref and Y_prior.
    mae_mean_n = mae_mean / mae_mean_prior
    _logger.info(f"Normalized MAE mean: {mae_mean_n}")
    
    # Non-linearly transform mae_mean_n.
    mae_mean_n_t = nonlinear_transform(mae_mean_n)
    _logger.info(f"Transformed Normalized MAE mean: {mae_mean_n_t}")
    
    ## Mean Absolute Error (MAE) of posterior standard deviation
    
    # Calculate the Mean Absolute Error (MAE) for the standard deviation of the datasets.
    # This measures the absolute difference between the standard deviations of the reference and compared datasets.
    mae_std = MAE(np.std(Y_ref, axis=0).ravel(), np.std(Y_ben, axis=0).ravel())
    _logger.info(f"MAE std: {mae_std}")
    
    # Calculate the MAE between the standard deviation of the reference dataset and the standard deviation of the prior dataset.
    mae_std_prior = MAE(np.std(Y_ref, axis=0).ravel(), np.std(Y_prior, axis=0).ravel())
    _logger.info(f"MAE std prior: {mae_std_prior}")
   
    # Normalize MAE by dividing the calculated MAE of the datasets by the MAE with the prior dataset.
    # This helps to compare how different the standard deviations of Y_ref and Y_ben are compared to Y_ref and Y_prior.
    mae_std_n = mae_std / mae_std_prior
    _logger.info(f"Normalized MAE std: {mae_std_n}")
    
    # Non-linearly transform mae_std_n.
    mae_std_n_t = nonlinear_transform(mae_std_n)
    _logger.info(f"Transformed Normalized MAE std: {mae_std_n_t}")

    ## Average ensemble standard deviation (AESD) 
    
    # Calculate the Average ensemble standard deviation (AESD) for the standard deviation of the datasets.
    # This measures the average posterior standard deviations in the datasets.
#     aesd_ref = AESD(np.std(Y_ref, axis=0).ravel())
#     _logger.info(f"AESD reference: {aesd_ref}")
    
#     aesd_rep = AESD(np.std(Y_ben, axis=0).ravel())
#     _logger.info(f"AESD representation: {aesd_rep}")
   
#     # Calculate the AESD for the standard deviation of the prior dataset.
#     aesd_prior = AESD(np.std(Y_prior, axis=0).ravel())
#     _logger.info(f"AESD prior: {aesd_prior}")
   
#     # Normalize AESD by dividing the difference between aesd_rep and aesd_ref by the difference between aesd_prior and aesd_ref.
#     # This helps to compare how the spectral density of Y_ben differs from Y_ref in comparison to Y_prior.
#     aesd_n = np.abs(aesd_rep - aesd_ref) / np.abs(aesd_prior - aesd_ref)
#     _logger.info(f"Normalized AESD: {aesd_n}")

#     # Non-linearly transform aesd_n.
#     aesd_n_t = nonlinear_transform(aesd_n)
#     _logger.info(f"Transformed Normalized AESD: {aesd_n_t}")

    ## Kolmogorov-Smirnov (KS)
    
    # Calculate the Kolmogorov-Smirnov (KS) distance between each pair of distributions (a and b).
    ksdist = np.zeros([np.shape(Y_ref)[1], np.shape(Y_ref)[2]], dtype=float)
    for j in range(np.shape(Y_ref)[2]):
        for i in range(np.shape(Y_ref)[1]):
            ksdist[i, j], _ = sp.stats.ks_2samp(Y_ref[::10, i, j], Y_ben[:, i, j])

    # Calculate the average KS distance across all pairs of distributions.
    ksdist_mean = np.mean(ksdist)
    _logger.info(f"KS distance space average: {ksdist_mean}")

    # Energy Distance
    
    # Calculate the Energy Distance (engdist) between two datasets (a and b).
    A = sp.spatial.distance.cdist(Y_ref.reshape(Y_ref.shape[0], -1)[::10], Y_ben.reshape(Y_ben.shape[0], -1)).mean()
    B = sp.spatial.distance.pdist(Y_ref.reshape(Y_ref.shape[0], -1)[::10]).mean()
    C = sp.spatial.distance.pdist(Y_ben.reshape(Y_ben.shape[0], -1)).mean()
    engdist = (2 * A - B - C) ** 0.5
    _logger.info(f"Energy distance: {engdist}")
 
    # Energy distance between the reference and the prior
    A_prior = sp.spatial.distance.cdist(Y_ref.reshape(Y_ref.shape[0], -1)[::10], Y_prior.reshape(Y_prior.shape[0], -1)).mean()
    C_prior = sp.spatial.distance.pdist(Y_prior.reshape(Y_prior.shape[0], -1)).mean()
    engdist_prior = (2 * A_prior - B - C_prior)
    _logger.info(f"Energy distance prior: {engdist_prior}")
    
    # Normalize the Energy Distance (engdist) by dividing it by the energy distance with the prior dataset (engdist_prior).
    engdist_n = engdist / engdist_prior
    _logger.info(f"Normalized Energy distance: {engdist_n}")

    # Non-linearly transform n_engdist if necessary (Option 1).
    engdist_n_t = nonlinear_transform(engdist_n)
    _logger.info(f"Transformed Normalized Energy distance: {engdist_n_t}")
    
    # Alternative normalization option (Option 2).
    engdist_n_2 = engdist / (2 * A) ** 0.5
    _logger.info(f"Another Normalized Energy distance: {engdist_n_2}")

    ## Prepare the comparison results in a DataFrame.
    Metrics = ["$NMAE^*$", "$NMAE^*$ std", "$NRMSE^*$", "$NRMSE^*$ std", "$AD_{KS}$", "$ND_E$"]
    # Metrics = ["MAE mean", "MAE std", "RMSE mean", "RMSE std", "KS distance", "Energy distance"]
    data= {
        "Original": [mae_mean, mae_std, rmse_mean, rmse_std, ksdist_mean, engdist],
        "Normalized": [mae_mean_n, mae_std_n, rmse_mean_n, rmse_std_n, ksdist_mean, engdist_n_2], 
        "TNormalized": [mae_mean_n_t, mae_std_n_t, rmse_mean_n_t, rmse_std_n_t, ksdist_mean, engdist_n_2]
    }
    # Return the comparison results DataFrame
    return pd.DataFrame(data, index=Metrics)



class DownloadBackground(object):
    def __init__(self, datadir, DOI,URL, KEY,interval=1):
        self.datadir = datadir
        self.URL = str(URL)
        self.DOI = str(DOI)
        self.KEY = str(KEY)

    def download_reference_data(self):
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
            
        try:
            _logger.debug("Start downloading. This may take a while.")
            os.environ["DATAVERSE_API_TOKEN"] = self.KEY
            Dataset.from_dataverse_doi(
                    doi=self.DOI,
                    dataverse_url=self.URL,
                    filedir = self.datadir,
                    api_token = self.KEY,
                    )
            
        except Exception:
            _logger.info(traceback.format_exc())

    def run(self):
        self.download_reference_data()

            
class BayesInvBench(object):
    """BayesInvBench, class that creates one pairwise comparison of sample comparison.

    Please make sure, that the base_url and the api key (if needed) are preset before.
    Args:
        DOI (string): standard 'doi:10.18419/darus-2382', needs to be the data location (on daRUS)
    """
    def __init__(self,config_file_path='config/config.yaml'):
        self.config_file_path = config_file_path
        # Read a few things from the config file.
        self.config = self.load_config()
        self.DOI = self.get_dataset_doi()
        self.API_KEY = self.get_api_key()
        self.scenario_name = self.get_scenario_name()
        self.current_dir = pathlib.Path(__file__)
        self.datapath = self.get_datapath()
        
        self.reference_path = None
        self.reference_chain_file_paths = None
        self.references_chains_data = None
        self.replication_data = None
        self.replication_name = self.get_replication_name()
        self.replication_piezometric_head = None
        self.bench_df = pd.DataFrame()
        # self.set_current_dir_to_current_dir()

    def get_api_key(self):
        key= self.config.get('api_key', None)
        _logger.info(f"API key: {key}")
        return key
        
    def load_config(self):
        with open(self.config_file_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        return config

    def get_datapath(self):
        datapath_r = self.config.get('datapath', './data')
        return self.current_dir.parent.parent.joinpath(datapath_r)

    def get_replication_name(self):
        return self.config.get('replication_name',None)
    
    def get_scenario_name(self):
        return self.config.get('scenario_name', None)

    def get_dataset_doi(self):
        return self.config.get('dataset_doi', None)

    def download_data_in_background(self):
        DownloadBackground(datadir=str(self.datapath),
                           DOI=self.DOI,
                           URL="https://darus.uni-stuttgart.de",
                           KEY=self.API_KEY).run()
        _logger.info(f"Started download of data to {self.datapath}")

    def set_reference_path(self):
        self.reference_path  = self.find_folder_scenarios(foldername="data/", prefix="ref_"+self.scenario_name)[0]

    def set_replication_name(self,replication_name):
        self.replication_name = replication_name

    def set_scenario_name(self,scenario_name):
        self.scenario_name = scenario_name

    def set_reference_chain_file_paths(self):
        if self.reference_path is None:
            self.set_reference_path()
        # self.reference_chain_file_paths = self.find_folder_filetype_files(pathlib.Path.joinpath(self.current_dir.parent,"data/",self.reference_path), ".h5")
        # self.reference_chain_file_paths = self.find_folder_filetype_files(self.reference_path, ".h5")
        self.reference_chain_file_paths = self.find_files_type(self.reference_path, "posterior_lnK")

    def find_files_type(self, foldername, Type="posterior_lnK"):
        
        files_type = []
        _logger.info(foldername)

        for root, dirs, files in os.walk(foldername, topdown=False):
            _logger.info(root)

            for name in files:
                _logger.info(name)
                if Type in name:
                    files_type.append(os.path.join(root, name))
        
        _logger.info(f"Found {len(files_type)} files: " + str(files_type))
        return files_type


    def find_folder_filetype_files(self,foldername="data/", filetypeabbr="h5"):
        # current_dir = self.current_dir.parent.parent
        # Search for h5 files
        matching_files = []
        # cwd = os.getcwd()
        _logger.info(foldername)
        pattern = os.path.join(foldername, f"*.{filetypeabbr}")
        normalized_path = os.path.normpath(pattern)
        forward_slash_path = normalized_path.replace(os.sep, '/')
        print(forward_slash_path)
        matching_files = glob.glob(forward_slash_path)
        _logger.info(f"Found {len(matching_files)} files: " + str(matching_files))
        return matching_files


    def find_folder_scenarios(self,foldername="data/", prefix="ref_"):
        # Parse for h5 files
        # current_dir = self.current_dir
        current_dir = pathlib.PurePath(inspect.getfile(BayesInvBench)).parent.parent
        foldernames_scenarios = []
        for root, dirs, files, in os.walk(pathlib.Path.joinpath(
                current_dir, foldername), topdown=False):
            for name in dirs:
                if name.startswith(prefix):
                    foldernames_scenarios.append(os.path.join(root, name))
        # _logger.info(f"Found {len(foldernames_scenarios)} files: " + str(foldernames_scenarios))
        return foldernames_scenarios


    def __mmap_h5(self, path, h5path):
        # Copied from https://github.com/kwikteam/klusta/blob/408e898e8d5dd1788841d1f682e51d0dc003a296/klusta/kwik/h5.py
        # BSD 3
        with h5py.File(path) as f:
            ds = f[h5path]
            # We get the dataset address in the HDF5 fiel.
            offset = ds.id.get_offset()
            # We ensure we have a non-compressed contiguous array.
            assert ds.chunks is None
            assert ds.compression is None
            assert offset > 0
            dtype = ds.dtype
            shape = ds.shape
        arr = np.memmap(path, mode='r', shape=shape, offset=offset, dtype=float)
        return arr


    def prepare_reference_data(self):
        
        if self.reference_chain_file_paths is None:
            self.set_reference_chain_file_paths()

        self.references_chains_data = [] # all chains
        self.references_chain1_data = [] # first chain
        self.references_chain2_data = [] # second chain
        for onechain in self.reference_chain_file_paths:
            _logger.info(onechain)
            # ln_cond = np.array(self.__mmap_h5(onechain, '/samples'))
            ln_cond = self.__mmap_h5(onechain, '/samples') # N x 100*100
            # reference data is generated with Matlab
            ln_cond_resh = np.reshape(ln_cond, (-1, 100, 100), order='F')
            # ln_cond_resh = ln_cond
            self.references_chains_data.append(ln_cond_resh)
            if "chain1" in pathlib.PurePath(onechain).name:
                self.references_chain1_data.append(ln_cond_resh)
            elif "chain2" in pathlib.PurePath(onechain).name:
                self.references_chain2_data.append(ln_cond_resh)
            _logger.debug("Put in another chain.")

    def load_replication_data(self):
        """
        load_replication_data
        ----------
        """

        replication_data_path = pathlib.Path.joinpath(self.datapath,"replication/EnKF_xu_2020/",self.replication_name).resolve()
        _logger.info(replication_data_path)
        repfile = self.find_folder_filetype_files(replication_data_path,"out")
        if len(repfile) == 0:
            _logger.warn("No replication file found!")
        else:
            repfile = repfile[0]
        # repfile = self.find_files_type(replication_data_path,"posterior_lnK")[0]
        lines = loadtxt(repfile, comments="#", delimiter=",", unpack=False,skiprows=3)
        # replication data (EnKF) is stored in C style
        self.replication_data = np.reshape(lines,(-1,100,100)) - 2.5 # original data has mean 0
        _logger.info("Load replication file. Assumed there is only one file.")

    def load_replication_piezometric_head(self):        
        rep_piezo_path = pathlib.Path.joinpath(self.datapath,"replication/EnKF_xu_2020/","piezometric_heads")
        _logger.info(rep_piezo_path)
        repfile = self.find_folder_filetype_files(rep_piezo_path,"out")[0]
        # if repfile.endswith("100.out"):
        if "100" in repfile:
            _logger.info("Found the ensemble 100 file")
        
        lines = loadtxt(repfile, comments="#", delimiter=",", unpack=False, skiprows=3)
        
        self.replication_piezometric_head = np.reshape(lines,(-1,100,100))
        _logger.info("Load Piezometric head file. Assumed there is only one file.")
        
        
    def plot_rep_mean_field(self):
        if self.replication_data is None:
            self.load_replication_data()

        parula = matplotlib_to_plotly(parula_map,255) 
        ln_cond_mean_shaped = self.replication_data.mean(axis=0)   
        title = "mean lnK Rep"
        f = px.imshow(ln_cond_mean_shaped, color_continuous_scale=parula, 
                      origin="lower", zmin=-4, zmax=2, title=title,
                      x=np.arange(0,5000,50), y=np.arange(0,5000,50))
        return f
        
    def __plot_mean_field(self,ln_cond_mean,leg="ln(K)"):
        """Plot e.g. 100x100 mean conductivity field.

        plot_mean_field
        ----------

        """
        # plt.grid(None)
        # ln_cond_mean_shaped = ln_cond_mean.reshape((100,100))
        # f = plt.imshow(((np.rot90(ln_cond_mean_shaped, 1))),
                    # interpolation="nearest",
                    # cmap=parula_map)  # plt.cm.jet
        
        ln_cond_mean_shaped = ln_cond_mean         
        parula = matplotlib_to_plotly(parula_map,255)
        title = "mean lnK Ref"
        f= px.imshow(ln_cond_mean_shaped, color_continuous_scale=parula, 
                     origin="lower", zmin=-4, zmax=2, title=title,
                     x=np.arange(0,5000,50), y=np.arange(0,5000,50))
        # Set the x-axis and y-axis labels
        f.update_xaxes(title_text="Easting")
        f.update_yaxes(title_text="Northing")

        # Set the colorbar title
        f.update_layout(coloraxis_colorbar=dict(title=leg))
        
        return f

    def plot_ref_mean_field(self):
        
        import dask.array as da
        # _logger.info(self.references_chains_data)
        _logger.info("Plot the first chain image.")
        # first chain after burn
        chain1_burn_in = self.burn_in()[0] 
        
        ## https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
        chain1_burn_in_con = da.concatenate(chain1_burn_in)
        return self.__plot_mean_field(chain1_burn_in_con.mean(axis=0))
            
    def plot_rep_mean_piezo_field(self):

        if self.replication_piezometric_head is None:
            self.load_replication_piezometric_head()
            
        # _logger.info(self.references_chains_data)
        # first chain after burn
        mean_head = self.replication_piezometric_head.mean(axis=0) 

        parula = matplotlib_to_plotly(parula_map,255)   
        title = "mean Head Rep"
        f = px.imshow(mean_head, color_continuous_scale=parula, 
                      origin="lower", zmin=0, zmax=20, title=title,
                      x=np.arange(0,5000,50), y=np.arange(0,5000,50))     
        
        return f
        
    def burn_in(self):
        """
        keep the samples after burn-in
        """
        if self.references_chains_data is None:
            self.prepare_reference_data()
            
        chains = [self.references_chain1_data, self.references_chain2_data]
        chains_burn_in = [[], []]
        for j in range(2):
            n_samples = np.zeros(len(chains[j]), dtype=int)
            for ind, data in enumerate(chains[j]):
                n_samples[ind] = data.shape[0]

            num_burn_in = n_samples.sum() // 2  # number of burn_in， treat first half as burn in
            ind_bool = n_samples.cumsum() <= num_burn_in
            start = num_burn_in - n_samples[ind_bool].sum()

            start_ind = True
            for ind, data in enumerate(chains[j]):
                if (ind_bool[ind] == False) and (start_ind == True):
                    chains_burn_in[j].append(data[start:])
                    start_ind = False
                elif ind_bool[ind] == False:
                    chains_burn_in[j].append(data)

        return chains_burn_in

    def plot_stem_from_table(self,ax,table,name="To be set."):
        # Sample data
        categories = table.index.tolist()
        values = np.round(table['TNormalized'].tolist(),3)
        print(values)
        # Create a figure and axis with a MATLAB-compatible theme
        plt.style.use('seaborn-v0_8-ticks')  # Apply MATLAB-like theme
        # fig, ax = plt.subplots(figsize=(8, 4))

        # Plot the stem plot

        markerline, stemlines, baseline = ax.stem(categories, values, markerfmt='x', linefmt = '--',label=name)
        # Customize marker properties
        markerline.set_markerfacecolor('b')
        markerline.set_markeredgecolor('b')
        markerline.set_markersize(8)
        # Sample data
        # Set y-axis limits
        # Set gridlines
        ax.grid(True, linestyle='--', alpha=0.7)  # Add gridlines with dashed style
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories, fontsize=11)  # Change 'sans-serif' to a font family available on your system
        ax.set_yticks(np.round(np.arange(0, 1.1, 0.1), 2))
        ax.set_yticklabels(np.round(np.arange(0, 1.1, 0.1), 2), fontsize=11)  # Adjust fontsize for ytick labels
  
        # Adjust legend label font size
        ax.legend(fontsize=11)
        # ax.set_ylim(0, 1.1)
        # Save the figure as a PDF file
        return ax

    def get_benchmark_table(self):
        _logger.info("This will take some time (5 min?), so maybe get a coffee.")
        references_chains_data_burn_in = self.burn_in()
        Y_ref = np.concatenate(references_chains_data_burn_in[0], axis=0) # chain 1
        # ind = np.random.randint(Y_ref.shape[0], size=10)
        # Y_ref_flat = Y_ref[ind].ravel()
        if self.replication_data is None:
            self.load_replication_data()
        Y_ben = self.replication_data 
        # ind = np.random.randint(Y_ben.shape[0], size=10)
        # Y_ben_flat = Y_ben[ind].ravel()
        
        priorS2 = pathlib.Path.joinpath(self.datapath, "reference/pCN-PT MCMC_xu_2020/prior_S2/prior_S2.out")
        
        priorS0_S1_S3_S4 = pathlib.Path.joinpath(self.datapath,"reference/pCN-PT MCMC_xu_2020/prior_S0_S1_S3_S4/prior_S0_S1_S3_S4.out") # Change the file name for S2
                                        
        prior_file = priorS2
                                        
        if "S2" not in self.scenario_name:
            prior_file = priorS0_S1_S3_S4
                                        
        Y_prior = loadtxt(prior_file, skiprows=3).reshape(-1, 100, 100) - 2.5 # original data has mean 0
        if self.bench_df.empty:
            
            table = compare_distributions_props(Y_ref,Y_ben,Y_prior)
            self.bench_df = table
        else:
            
            table = self.bench_df
        return table
    
    def r_hat(self):
        """
        Calculate the potential scale reduction factor
        """
        import dask.array as da
        
        chains_burn_in = self.burn_in()
        for ind, chain in enumerate(chains_burn_in):
            ## https://stackoverflow.com/questions/13780907/is-it-possible-to-np-concatenate-memory-mapped-files
            chains_burn_in[ind] = da.concatenate(chain, axis=0)
        
        R = PSRF(chains_burn_in).compute()
        return R
    
    def make_comparison_plot(self):
        """
        Plot reference data and candidate data
        Get the spider plot
        """
        references_chains_data_burn_in = self.burn_in()
        Y_ref = np.concatenate(references_chains_data_burn_in[0], axis=0) # chain 1
        # ind = np.random.randint(Y_ref.shape[0], size=10)
        # Y_ref_flat = Y_ref[ind].ravel()
        if self.replication_data is None:
            self.load_replication_data() 
        Y_ben = self.replication_data 
        # ind = np.random.randint(Y_ben.shape[0], size=10)
        # Y_ben_flat = Y_ben[ind].ravel()
        # plt.rcParams['axes.spines.right'] = False
        # plt.rcParams['axes.spines.top'] = False
        plt.rcParams['xtick.labelsize'] = 12 #22                               # Set global parameters for the plot
        plt.rcParams['ytick.labelsize'] = 12 #22
        plt.rcParams['axes.labelsize'] = 12 #22
        plt.rcParams['axes.titlesize'] = 14 #30
        # plt.grid(None)
        
        # fig1, f1_axes = plt.subplots(nrows=3,ncols=2,figsize=[3*11.69/3,2*9.27/3],layout="constrained", dpi=300)
        fig1, f1_axes = plt.subplots(nrows=3, ncols=2, figsize=(2*4, 3*4), dpi=300) 
        # spec = fig1.add_gridspec(3, 2)
        # Mean lnK
        Y_ref_mean = np.mean(Y_ref, axis=0)
        # f00 = f1_axes[0,0].imshow(np.rot90(np.mean(Y_ref,axis=0),1),interpolation="bilinear", cmap=parula_map)
        f00 = f1_axes[0,0].imshow(Y_ref_mean, interpolation="bilinear", cmap=parula_map, 
                                  origin="lower", vmin=-4, vmax=2, extent=(0,5000,0,5000))
        f1_axes[0,0].set_title('mean lnK Ref')
        f1_axes[0,0].set_xlabel('Easting')
        f1_axes[0,0].set_ylabel('Northing')
        add_colorbar(f00)
        # Std lnK
        Y_ref_std = np.std(Y_ref, axis=0)
        # f01 = f1_axes[0,1].imshow(np.rot90(np.std(Y_ref,axis=0),1),interpolation="bilinear", cmap=parula_map)
        f01 = f1_axes[0,1].imshow(Y_ref_std, interpolation="bilinear", cmap=parula_map, 
                                  origin="lower", vmin=0, vmax=2.5, extent=(0,5000,0,5000))
        f1_axes[0,1].set_title('std lnK Ref')
        f1_axes[0,1].set_xlabel('Easting')
        f1_axes[0,1].set_ylabel('Northing')
        add_colorbar(f01)
        # Mean Comp
        Y_ben_mean = np.mean(Y_ben, axis=0)
        # f02 = f1_axes[0,2].imshow(np.rot90(np.mean(Y_ben,axis=0),1),interpolation="bilinear", cmap=parula_map)
        f02 = f1_axes[1,0].imshow(Y_ben_mean, interpolation="bilinear", cmap=parula_map, 
                                  origin="lower", vmin=-4, vmax=2, extent=(0,5000,0,5000))
        f1_axes[1,0].set_title('mean lnK Rep')
        f1_axes[1,0].set_xlabel('Easting')
        f1_axes[1,0].set_ylabel('Northing')
        add_colorbar(f02)
        # Std Comp
        Y_ben_std = np.std(Y_ben, axis=0)
        # f03 = f1_axes[0,3].imshow(np.rot90(np.std(Y_ben,axis=0),1),interpolation="bilinear", cmap=parula_map)
        f03 = f1_axes[1,1].imshow(Y_ben_std, interpolation="bilinear", cmap=parula_map, 
                                  origin="lower", vmin=0, vmax=2.5, extent=(0,5000,0,5000))
        f1_axes[1,1].set_title('std lnK Rep')
        f1_axes[1,1].set_xlabel('Easting')
        f1_axes[1,1].set_ylabel('Northing')
        add_colorbar(f03)

        if self.bench_df.empty:
            self.get_benchmark_table()

        table = self.bench_df

        _logger.info("Wrote metrics into bench_df")
        # Create a new axes object spanning the two columns in the last row
        f1_axes[2,0].set_axis_off() # hide the original axis
        f1_axes[2,1].set_axis_off() 
        ax_combined = plt.subplot2grid((3, 2), (2, 0), colspan=2, fig=fig1)
        # Add your plot to the combined axes
        self.plot_stem_from_table(ax_combined, table, self.scenario_name)
        
        fig1.tight_layout()

        figure_path_r = self.config.get('figurepath')
        figure_path = self.current_dir.parent.parent.joinpath(figure_path_r)
        figure_path.mkdir(parents=True, exist_ok=True)

        fig1.savefig(f"{figure_path}/Comprison_{self.scenario_name}_{self.replication_name}.png", dpi=150)
        # _logger.info("Wrote png")
        fig1.savefig(f"{figure_path}/Comparison_{self.scenario_name}_{self.replication_name}.pdf", dpi=150)
        fig1.savefig(f"{figure_path}/Comparison_{self.scenario_name}_{self.replication_name}.eps", dpi=150)

        table_path_r = self.config.get('tablepath')
        table_path = self.current_dir.parent.parent.joinpath(table_path_r)
        table_path.mkdir(parents=True, exist_ok=True)
        table.to_csv(f"{table_path}/Comparison_{self.scenario_name}_{self.replication_name}.csv")
        return fig1, table



def parse_args(args):
    """Parse command line parameters

    Args:
    args (List[str]): command line parameters as list of strings
    (for example  ``["--help"]``).

    Returns:
    :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Geostatistical Inversion Benchmarking")

    parser.add_argument(
        "-s",
        "--scenario",
        dest="scenario",
        help=
        f"Choose a scenario from {[pathlib.PurePath(find_folder_scenarios()[i]).parts[-1] for i in range(len(find_folder_scenarios()))]}. If there are none available, first download the reference files with -d")

    parser.add_argument(
        "-ref",
        "--reference",
        dest="reference",
        help=f"Choose a reference from {[pathlib.PurePath(find_folder_scenarios('data/','rep_')[i]).parts[-1] for i in range(len(find_folder_scenarios('data/','rep_')))]}."
        )

    parser.add_argument("-d",
                        dest="dow",
                        help="Will download the reference files into ./data/*",
                        action="store_true")
    parser.add_argument("-cp",
                        dest="compareplot",
                        help="Will analyse a given scenario",
                        action="store_true")
    parser.add_argument("-a",
                        dest="analyse",
                        help="Will analyse a given scenario",
                        action="store_true")

    parser.add_argument("-pref",
                        dest="plotref",
                        help="Will plot the given reference scenario",
                        action="store_true")

    parser.add_argument(
        "--version",
        action="version",
        version="Geostatistical Inversion Benchmarking {ver}".format(
            ver=__version__),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
    loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel,
                        stream=sys.stdout,
                        format=logformat,
                        datefmt="%Y-%m-%d %H:%M:%S")


