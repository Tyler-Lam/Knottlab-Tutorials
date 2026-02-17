import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib import colors, gridspec
import numpy as np
import math
import time
import seaborn as sns
from tqdm import tqdm
import contextlib
import joblib
from joblib import Parallel, delayed, parallel_backend
from itertools import combinations, product
import warnings
from collections import defaultdict, Counter
from pathlib import Path
import os
import re
import multiprocessing
import traceback
from numba_progress import ProgressBar

from scipy import stats
from scipy.stats import beta, mannwhitneyu, false_discovery_control, chi2, lognorm, nbinom
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.special import betaln, gammaln
from scipy.optimize import minimize_scalar
from sklearn.metrics import silhouette_score

from patsy import dmatrices
from formulaic_contrasts import FormulaicContrasts

import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults
from statsmodels.othermod.betareg import BetaModel, BetaResults, BetaResultsWrapper
from statsmodels.discrete.discrete_model import NegativeBinomial, NegativeBinomialResults
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP, ZeroInflatedNegativeBinomialResults
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLMResults

from statannotations.Annotator import Annotator

import scanpy as sc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp

import gc

# Stackoverflow solution to make tqdm work with joblib.Parallel
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        

class GenericContrast:
    """
    Class to do generic manipulation of multiple contrast vectors by an operation. Stores the conditions
    and the operation for each contrast to be applied to a design matrix

    Parameters:
    ------------
    conditions: dict[dict]
        Set of conditions that define contrast vectors. Each key represents the name of
        a contrast vector
    op: func
        Function that takes in a set of contrast vectors and returns one contrast vector.
        By default, if 2 conditions are given, the operation takes a "test" and "ref" vector and calculates the difference
        If 1 is given, the contrast vector is returned for the given condition
    name: str
        Name for the given comparison represented by the contrast
    
    Example: To setup a contrast to measure the difference between Diagnosis stage A and B
    conditions = {
        'test': dict(Diagnosis = "A"),
        'ref': dict(Diagnosis = "B")
    }
    op = lambda test, ref: test - ref
    name = 'A_vs_B'
    
    Example: To setup a contrast to see if treatment has a different effect in groups A and B
    conditions = {
        'c1': dict(Diagnosis = "A", treatment = 0),
        'c2': dict(Diagnosis = "A", treatment = 1)
        'c3': dict(Diagnosis = "B", treatment = 0),
        'c4': dict(Diagnosis = "B", treatment = 1),
    }
    op = lambda c1, c2, c3, c4: (c2 - c1) - (c4 - c3)
    name = "A_vs_B___treatment_interaction"
    
    """
    def __init__(self, conditions, op = None, name = ""):
        self.conditions = conditions
        self.op = op
        # Setup some default operations if none are given
        if op is None:
            # If only one condition, return contrast vector for that condition
            if len(conditions) == 1:
                self.op = lambda **kwargs: next(iter(kwargs.values()))
            # If two conditions, assume one is 'test' and one is 'ref'
            if len(conditions) == 2:
                self.op = lambda test, ref: test - ref
        self.name = name

# Log(n choose k) for stability in log-likelihood functions
def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def get_celltype_annot_region_feature_type(
    col,
    annot_dict = {'2': 'Stroma', '3': 'Epithelium', '0': 'All Tissue'},
):
    """
    Given a column name, return the cell type, annotation region, and feature type. This assumes columns formatted according to the tutorial
    Feature types that this can return are:
    * {cell_category}_proportion
    * {cell_category}_density
    * {child_category}_per_{parent_celltype}
    * {cell_category}_spatial_correlation_central_cell_{central_celltype}
    * {cell_category}_covering_fraction_{celltype_1}
    where {cell_category} is primary_celltype, secondary_celltype, etc.
    
    Parameters:
    -----------
    col: str
        Column name to parse
    annot_dict: dict
        Dictionary mapping annotation mask values to region names
    """
    
    region = col.split('annot_region_')[1][0] if 'annot_region' in col else '0'
    celltype = col.split('___')[-1]
    subtype_pattern = re.compile(f".*proportion_per_.*_per_annot_region_.*")
    feature_type = ''
    if subtype_pattern.match(col):
        feature_type = col.split('_per_annot_region')[0]
    elif 'proportion' in col:
        feature_type = col.split('proportion')[0] + 'proportion'
    elif 'density' in col:
        feature_type = col.split('density')[0] + 'density'
    elif 'spatial_correlation' in col:
        feature_type = col.split('_radius')[0]
    elif 'covering_fraction' in col:
        feature_type = col.split('_radius')[0]
    return celltype, annot_dict[region], feature_type

# Given a p-value, get the size prop to -log(p)
def get_size_from_pval(pval, smin = 80, smax = 500, pmin = 0.05):
    size = -np.log10(max(pval, pmin))
    size_scaled = smin + size * (smax - smin) / (-np.log(pmin))
    return size_scaled

def make_dotplot(
    stat_df, feature_type = None, x_col = 'annot_region', y_col = 'celltype', 
    hue_col = 'mu', pval_col = 'p-wald', pval_threshold = 0.05, vmax = None,
    xlabel = None, ylabel = None, cbar_label = 'Effect Size', figsize = (6, 5), 
    test_group = 'Test', ref_group = 'Ref',
    title = "", save_as = None, show = True, fig = None):
    """
    Make a dotplot to do pairwise comparisons between classes
    
    Parameters:
    -------------
    df: pd.DataFrame
        Stats dataframe returned from GLMCollection
    x_col, y_col: str
        dataframe col names to place the row/column for each feature. Defaults to
        'annot_region' for x and 'celltype' for y
    hue_col: str
        Dataframe column to color each point. Defaults to mu (effect size)
    pval_col: str
        Dataframe column for pvalue (size + significance). Defaults to p-wald
    pval_threshold: str
        Minimum p-value to indicate feature is significant
    vmax: float | None
        Maximum abs(hue_col) for color normalization
    xlabel, ylabel: str
        Label for x/y-axis (defaults to x_col/y_col)
    cbar_label: str
        Label for the colorbar
    test_group, ref_group: str
        Labels for colorbar to indicate enrichment in test vs ref class
    title: str
        Title for the plot
    """
    with sns.axes_style('whitegrid'):
        
        # Filter to select features of a given type
        df = stat_df.copy()
        if feature_type is not None:
            df = df[df['feature_type'] == feature_type].copy()
        
        # Get the size of the points from the p-values
        size_scaled = df[pval_col].apply(lambda x: get_size_from_pval(x, pmin = pval_threshold))
        
        if fig is None:
            fig = plt.figure(figsize = figsize)
            
        # Define the grid for the plots/heatmaps
        gs = gridspec.GridSpec(1, 2, width_ratios = [8, 6], wspace = 0.1, figure = fig)
        
        # Define axis for the scatterplot
        ax0 = fig.add_subplot(gs[0])
        
        # Define the colorbar/legend gridspecs
        gs_cbar = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios = [1, 4], subplot_spec=gs[1], wspace = 2)
        ax1 = fig.add_subplot(gs_cbar[0]) # For the colorbar
        ax2 = fig.add_subplot(gs_cbar[1]) # For the pvalue-size legend
        ax2.axis('off')
        
        # vmax for normalizing colorbars
        vmax = max(np.abs(df[hue_col])) if vmax is None else vmax        
        if 'mu' in hue_col:
            norm = SymLogNorm(linthresh=0.001, vmin = -vmax, vmax = vmax, clip = True)
        else:
            norm = Normalize(vmin = -vmax, vmax = vmax, clip = True)

        # Draw the actual colorbar
        sns.scatterplot(
            x = x_col,
            y = y_col,
            hue = hue_col,
            data = df,
            palette = 'vlag',
            hue_norm = norm,
            size = size_scaled,
            sizes = (size_scaled.min(), size_scaled.max()),
            legend = None,
            edgecolor = 'black',
            ax = ax0,
        )
        
        # Loop over x/y positions in the scatterplot to put a marker on significant features
        x_pos = {val: pos for pos, val in enumerate(df[x_col].unique())}
        y_pos = {val: pos for pos, val in enumerate(df[y_col].unique())}
        for idx, row in df[df[pval_col] <= pval_threshold].iterrows():
            sns.scatterplot(
                x = [x_pos[row[x_col]]],
                y = [y_pos[row[y_col]]],
                hue = [row[hue_col]],
                palette = 'vlag',
                hue_norm = norm,
                size = [size_scaled.loc[idx]],
                sizes = (size_scaled.loc[idx], size_scaled.loc[idx]),
                legend = None,
                edgecolor = 'black',
                marker = r'$\odot$',
                linewidth = 0.75,
                ax = ax0
            )
        
        # Plot formatting
        ax0.set_xlim(-0.5, len(x_pos) - 0.5)
        
        if xlabel is not None:
            ax0.set_xlabel(xlabel)
        if ylabel is not None:
            ax0.set_ylabel(ylabel)
            
        # Make the colorbar and apply labels
        sm = plt.cm.ScalarMappable(cmap = 'vlag', norm = norm)
        sm.set_array([])
        cbar_mu = fig.colorbar(sm, cax = ax1)
        cbar_mu.set_label(cbar_label)
        cbar_mu.ax.text(0, -vmax * 1.04, f'Higher in {ref_group}', va = 'top', ha = 'left')
        cbar_mu.ax.text(0, vmax * 1.04, f'Higher in {test_group}', va = 'bottom', ha = 'left')

        # Use 1, 0.5, 0.2, 0.1, and 0.05 as the p-value legend size references
        # This is a janky way of doing it but it keeps sizes consistent between the legend/plot
        # For each value, add an empty scatterplot with the correct size/label
        # Then take the legend handles from this and make a new legend, which is drawn on the legend axis
        for p in [1, .5, .2, .1]:
            if p <= pval_threshold:
                continue
            size = get_size_from_pval(p, pmin = pval_threshold)
            ax0.scatter([], [], s = size, c = 'lightgray', edgecolor = 'black', label = f'p = {p}')
        ax0.scatter([], [], s = get_size_from_pval(pval_threshold, pmin = pval_threshold), c = 'lightgray', linewidth = 0.75, marker = r'$\odot$', edgecolor = 'black', label = fr'p $\leq$ {pval_threshold}')
        h, l = ax0.get_legend_handles_labels()
        ax2.legend(h, l, title = r'$\text{size}\propto-\log_{10}(p)$', markerscale = 1, frameon = False, loc = 'center left', labelspacing = 1.1)
        fig.suptitle(title)
        
        if save_as is not None:
            plt.savefig(f'{save_as}.png', bbox_inches = 'tight')
        if show:
            plt.show()
            return fig
        else:
            plt.close()
            


def get_progression_df(stat_res, ref_group, ref_key = 'ref_group', test_key = 'test_group'):
    """
    Given a statistics result from the GLMCollection, aggregate the dataframes to give feature effects relative to the reference group
    
    Parameters:
    ------------
    stat_res: pd.DataFrame
        Output from GLMCollection.run_stats() (or run_stats_with_permutations())
    ref_group: dict
        Dictionary defining the reference group predictors
    """
    res_df = pd.DataFrame()
    
    for key, res in stat_res.items():
        if key == 'llr':
            continue
        tmp_df = res['df'].copy()
        if res[ref_key] == ref_group:
            for col, val in res[test_key].items():
                tmp_df[col] = val
        elif res[test_key] == ref_group:
            for col, val in res[ref_key].items():
                tmp_df[col] = val
                
            tmp_df['effect'] = -1 * tmp_df['effect']
        
        else:
            continue

        res_df = pd.concat([res_df, tmp_df], axis = 0, join = 'outer')
    
    res_df = res_df.reset_index()
    res_df = res_df.rename(columns = {'index': 'feature'})
    return res_df

def plot_progression_heatmap(
    stat_df, llr_df = None, index = ['Celltype', 'Region'], columns = 'Diagnosis', values = 'effect', signif_col = None,
    ncols = 1, vmax = None, title = "", xlabel = None, cbar_label = 'log2FoldChange', row_cluster = True,
    pval_col = 'p-nom', cbar_width = 0.03, cbar_height = 0.45, order = None, figsize = (10,10), show = True, save_dir = None):
    """
    Make the heatmap for a feature across several values of a predictor. 
    Makes a pivot table and pass to seaborn clustermap and does additional formatting
    I don't know what seaborn does but making adjustments (like placing legends and padding) are very hard to do from the clustermap
    
    Parameters:
    -----------
    stat_df: pd.DataFrame
        Dataframe obtained from get_progression_df
    llr_df: pd.DataFrame
        Dataframe of Log likelihood ratio test results (used to label features using FDR q-value)
    index: list[str]
        Columns used to index the pivot table
    columns: str
        Col name to use a columns for the pivot table
    values: str
        Values for the pivot table
    signif_col: str
        Column in llr_df to indicate significance. If none, use llr_df[pval_col] <= 0.05
    ncols: int
        Number of columns to use for celltype legend. (Values > 1 only really work for niches)
    vmax: float
        Maximum value for heatmap normalization
    title: str
        Title for the plot
    xlabel: str
        Label for the x-axis (defaults to "columns" parameter)
    cbar_label: str
        Label for the colorbar
    row_cluster: bool
        Cluster the rows and plot with a dendrogram. Otherwise order rows based on the index
    pval_col: str
        Column name for feature-wise p-value labels on clustermap
    cbar_width: float
        width of colorbar
    cbar_height: float
        maximum height of colorbar
    order: list[str]
        Left to right order for the columns
    """

    # Make the pivot table for the clustermap
    pivot = stat_df.pivot(index = index, columns = columns, values = values)
    pivot = pivot.dropna(axis = 0, how = 'all') # Remove rows with all nan values
    pivot = pivot.fillna(0)
    pivot = pivot.sort_index(level = range(len(index)))
    
    if order is None:
        order = pivot.columns.unique()
        
    if len(pivot) == 0:
        return
    elif len(pivot) == 1:
        row_cluster = False
        
    # Set colormaps depending on size of celltype classes
    ct_cmap = 'husl'
    if pivot.index.levels[0].size <= 10:
        ct_cmap = 'tab10'
    elif pivot.index.levels[0].size <= 20:
        ct_cmap = 'tab20'

    # Make the colormaps for each row
    row_cmap = {}
    row_colors = {}
    for i, idx in enumerate(index):
        if idx == 'Celltype':
            palette = sns.color_palette(ct_cmap, n_colors = pivot.index.levels[i].size)
            row_cmap[idx] = {ct: color for ct, color in zip(pivot.index.get_level_values(idx).unique().sort_values(), palette)}
            row_colors[idx] = pivot.index.get_level_values(idx).map(row_cmap[idx])
        elif idx == 'Region':
            palette = sns.color_palette('Set2', n_colors = pivot.index.levels[i].size)
            row_cmap[idx] = {ct: color for ct, color in zip(pivot.index.get_level_values(idx).unique().sort_values(), palette)}
            row_colors[idx] = pivot.index.get_level_values(idx).map(row_cmap[idx])
        elif idx == 'Radius':
            palette = sns.color_palette('flare', n_colors = pivot.index.levels[i].size)
            row_cmap[idx] = {ct: color for ct, color in zip(pivot.index.get_level_values(idx).unique().sort_values(), palette)}
            row_colors[idx] = pivot.index.get_level_values(idx).map(row_cmap[idx])
        else:
            palette = sns.color_palette('tab10', n_colors = pivot.index.levels[i].size)
            row_cmap[idx] = {ct: color for ct, color in zip(pivot.index.get_level_values(idx).unique().sort_values(), palette)}
            row_colors[idx] = pivot.index.get_level_values(idx).map(row_cmap[idx])
            
    if vmax is None:
        vmax = np.max(np.abs(pivot)) 
    else:
        vmax = min(vmax, np.max(np.abs(pivot)))
    
    # Draw the clustermap
    g = sns.clustermap(
        pivot[[c for c in order if c in pivot.columns]], col_cluster = False, row_cluster = row_cluster, # No dendrogram on rows/cols to keep things sorted by category
        cmap = 'vlag_r', vmin = -vmax, vmax = vmax,
        row_colors = pd.DataFrame({idx: row_colors[idx] for idx in index}, index = pivot.index),
        cbar_kws = {'label': cbar_label},
        figsize = figsize,
        dendrogram_ratio = (.2, 0.05)
    )
    
    # Get the row order for annotation placement
    if row_cluster:     
        row_order = pivot.index[g.dendrogram_row.reordered_ind]
    else:
        row_order = pivot.index
        
    # Formatting stuff
    g.figure.suptitle(title)
    g.ax_heatmap.set_ylabel('')
    if xlabel is not None:
        g.ax_heatmap.set_xlabel(xlabel)
    
    # Set y-ticks to be global FDR q-values
    if llr_df is not None:
        llr_df = llr_df.set_index(index)
        ordered_llr_df = llr_df.loc[row_order]
        #ordered_llr_df = pivot.merge(llr_df['p-adj'], on = index, how = 'left')
        ylabels = ordered_llr_df[pval_col].values
        g.ax_heatmap.set_yticks([0.5 + x for x in range(len(pivot))])
        yticks = g.ax_heatmap.get_yticklabels()
        if signif_col is None:
            signif = llr_df.loc[row_order, pval_col] <= 0.05
        else:
            signif = llr_df.loc[row_order, signif_col]
        for label, tick, sig in zip(ylabels, yticks, signif):
            tick.set_text(f'p={label:.3f}')                
            if sig:
                tick.set_fontweight('bold')
        g.ax_heatmap.set_yticklabels(yticks)
            

    # Annotate each box in the heatmap with the corresponding value
    for i, (idx, row) in enumerate(pivot.loc[row_order, [c for c in order if c in pivot.columns]].iterrows()):
        for j, col in enumerate(pivot.loc[row_order, [c for c in order if c in pivot.columns]].columns):
            if np.isnan(row[col]):
                continue
            g.ax_heatmap.text(
                j + 0.5, i + 0.5,
                fr'$\Delta$={row[col]:.3f}',
                ha = 'center', va = 'center'
            )
            
    # Plot the legends for each index
    # Initial position = top of heatmap
    pos = g.ax_heatmap.get_position()
    y1 = pos.y1
    legs = {} # Dictionary to keep legend for each index evel
    for idx in index:

        # Add handles from the row colormaps
        handles = [Patch(color = c, label = l) for l, c in row_cmap[idx].items()]
        if idx == 'Celltype':
            # This is my jank fix to insert newlines instead of underscores for celltypes
            # Mainly for spacing cause otherwise the text overlaps into the heatmap
            handles = [Patch(color = c, label = l if isinstance(l, int) else l.replace('Epithelial_', 'Epithelial\n').replace("Epithelium_", "Epithelium\n")) for l, c in row_cmap[idx].items()]
        
        # Draw the legend to the right of the heatmap
        legs[idx] = g.cax.legend(
            handles = handles,
            title = 'Cell label' if idx == 'Celltype' else idx,
            title_fontproperties = {'weight': 'bold'},
            bbox_to_anchor = (g.ax_heatmap.get_position().x1 + 0.1, y1),
            loc = 'upper left',
            frameon = False,
            bbox_transform = g.figure.transFigure,
            alignment = 'left',
            ncol = ncols if idx == 'Celltype' else 1 
        )
        
        # Get the bottom coordinate of the legend as a reference for the next legend
        renderer = g.figure.canvas.get_renderer()
        bbox_leg = legs[idx].get_window_extent(renderer = renderer)
        bbox_leg_axes = bbox_leg.transformed(g.figure.transFigure.inverted())
        y1 = bbox_leg_axes.y0
        
    for l in legs:
        g.cax.add_artist(legs[l])
    
    # Move the heatmap so it's on the right of the heatmap and below the legends
    pos_heatmap = g.ax_heatmap.get_position()
    g.cax.set_position([pos_heatmap.x1 + 0.12, pos_heatmap.y0, cbar_width, min(cbar_height, y1 - pos_heatmap.y0 - 0.03)])
    # Label colorbar
    g.cax.text(0, vmax+.01, 'Higher in test group', va = 'bottom', ha = 'left')
    g.cax.text(0, -vmax-.01, 'Higher in ref group', va = 'top', ha = 'left')
    
    if save_dir is not None:
        plt.savefig(f'{save_dir}.png', bbox_inches = 'tight')
        plt.close()
    if show:
        plt.show()
        return g
    else:
        plt.close()

def get_empirical_pvalues(df, perm_df, stat_col = 'stat', col_to_add = 'p-nom'):
    """
    Fast way to get empirical p-values for multiple features from permutations. Adds the nominal p-values
    in place and returns the dataframe
    
    Parameters:
    ------------
    df: pd.DataFrame
        Dataframe indexed by unique feature name with column for test statistics
    perm_df: pd.DataFrame
        Dataframe for each permutation, also indexed by feature with column for test statistics
    stat_col: str
        Name of column containing test statistic
    col_to_add: str
        Name of column to add containing empirical p-values
    """
    
    # Mask out nans in stat dataframe and mask those same features in the permutations
    # Also mask nans in the permutation dataframe
    mask = ~df[stat_col].isna()
    perm_mask = (~perm_df[stat_col].isna()) & (perm_df.index.isin(mask[mask].index))
    
    sub_perm_df = perm_df[perm_mask]
    # Sort the feature names and get the index map using np.unique
    # obs_idx maps obs_feat back to the original order
    obs_feat, obs_idx = np.unique(mask[mask].index, return_inverse = True)
    
    # For each permutated feature, get the index of the original feature
    # in the sorted order
    perm_idx = np.searchsorted(obs_feat, sub_perm_df.index)
    
    # For each permuted feature, get the observed test statistic
    obs_vals = df.loc[obs_feat, stat_col].values
    obs_perm_vals = obs_vals[perm_idx]    
    
    # Boolean map to see if permuted statistic is greater than the observed
    geq = sub_perm_df[stat_col].values >= obs_perm_vals
    
    # Get the number of times a feature has a greater permuted statistic
    # by counting the index of features using the geq mask
    counts = np.bincount(perm_idx, weights = geq)
    total = np.bincount(perm_idx)
    # Calculate the p-values (ordered by the sorted features) with a +1 correction for zero values
    pvals = (counts + 1) / (total + 1)
    df.loc[obs_feat, col_to_add] = pvals
    return df

# Correlation calculations to handle NaN values in permutations
# Using numba loops is much faster than scipy.spearmanr for each pair of features
from numba import njit, prange
@njit(parallel = True)
def fast_corr(X, pbar = None):
    n, m = X.shape
    res = np.empty((m, m), dtype = np.float32)
    
    # Loop over pairs of rows
    for i in prange(m):
        res[i, i] = 1.0
        for j in range(i+1, m):
            
            # Calculate means of each column (omitting nans)
            count = 0
            xi = 0.0
            xj = 0.0
            for k in range(n):
                if not np.isnan(X[k, i]) and not np.isnan(X[k,j]):
                    xi += X[k,i]
                    xj += X[k,j]
                    count += 1
            if count < 2:
                res[i, j] = np.nan
                res[j, i] = np.nan
            mean_i = xi / count
            mean_j = xj / count

            # Calculate variance and covariance
            cov = 0.0
            var_i = 0.0
            var_j = 0.0
            
            for k in range(n):
                # Skip nans
                if not np.isnan(X[k,i]) and not np.isnan(X[k,j]):
                    di = X[k,i] - mean_i
                    dj = X[k,j] - mean_j
                    cov += di * dj
                    var_i += di*di
                    var_j += dj*dj
            if var_i == 0 or var_j == 0:
                corr = np.nan
            else:
                corr = cov / np.sqrt(var_i * var_j)
            res[i,j] = corr
            res[j,i] = corr
            
        if pbar is not None:
            pbar.update(1)
    return res

# Calculate the simes p-value given a pandas series of p-values
def get_simes_p(pvals):
    return min(false_discovery_control(pvals))

# Used for applying fdr control to groups of a dataframe
def apply_fdr(df, pval_col = 'p-nom', out_col = 'cluster-p-adj'):
    df[out_col] = false_discovery_control(df[pval_col])
    return df

# Run Benjamini Bogomolov selection by clustering based on permutations
def run_fdr_corrections(perm_df, llr_df, alpha = 0.05, plot_threshold = False, stat_col = 'stat', pval_col = 'p-nom', nan_behavior = 'omit', n_jobs = None, silhouette_batch_size = None, random_state = None):
    """
    Run benjamini bogomolov FDR correction by clustering features by spearman correlation across permutations
    
    Parameters:
    ------------
    perm_df: pd.DataFrame
        permutation dataframe. Must have a 'feature', 'stat', and 'perm_iter' column
    llr_df: pd.DataFRame
        Log likelihood-ratio dataframe
    alpha: float
        significance threshold, defaults to 0.05
    plot_threshold: bool
        Plot silhouette score vs threshold for clustering
    stat_col: str
        Column for test statistic
    nan_behavior: 'omit' | 'zero'
        How nans are handled in the correlation matrix. Must be 'omit' or 'zero'.
        "omit" removes features (starting from highest nan counts) until no nans remain
        "zero" replaces nans with 0. This is much faster, but should only be used if you are sure the nans are due to lack of pairwise entries and are not expected to be correlated, not due to true 0 variance in the correlation
    n_jobs: int | None
        Number of jobs for calculating clustering threshold from silhouette scores.
        If None, default to ncpus - 1
    silhouette_batch_size: int | None
        Batch size for silhouette score calculation. If none, calculate on entire linkage matrix
    random_state: int | None
        Random seed for batch sampling of silhouette score
    """
    t_start = time.time()
    print("Running Benjamini-Bogomolov selection criteria based on permutation clusters")
    # Filter the permutation df to only use features in the llr df
    sub_perm_df = perm_df[perm_df.index.isin(llr_df.index)]
    
    t0 = time.time()
    print("Ranking test statistics ... ", end = "")
    # Make the pivot table and rank dataframe
    pivot = sub_perm_df.reset_index().pivot(index = 'perm_iter', columns = sub_perm_df.index.name, values = stat_col)
    ranks = pivot.rank(axis = 0)
    del pivot
    gc.collect()
    print(f"done: {(time.time() - t0)/60:.2f} min")
    
    # Make the correlation matrix and dataframe
    # Use numba progress bar to track progress
    with ProgressBar(total=ranks.shape[1], desc = "Calculating correlation matrix") as progress:
        corr = fast_corr(ranks.to_numpy(dtype = np.float32), pbar = progress)
    # Downcast to float16 for memeory since these matrices can be large
    corr = corr.astype(np.float16)
    
    # Define the mask of features to keep outside the nan corrections
    mask_to_keep = np.ones(len(corr), bool)
    
    if np.isnan(corr).any():
        print("Removing features with nans")
        if nan_behavior == 'zero':
            corr[np.isnan(corr)] = 0
        elif nan_behavior == 'omit':
            # Find features with highest nan counts
            print('   Creating mask for nan values ... ', end = "")
            t0 = time.time()
            # Since matrix is symmetric, look for nans above the diagonal
            upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
            # Get all indices with a nan in the upper diagonal
            mask = np.isnan(corr) & upper_mask
            total_nans = len(np.unique(np.concatenate(np.where(mask))))
            print(f'done: {(time.time() - t0) / 60:.2f} min')
            feats_to_remove = []
            idx_to_remove = []
            # Remove row/col with the most nans iteratively until none remain
            print('   Finding features to remove ... ', end = '')
            t0 = time.time()
            while mask.any():
                # Get the number of times a feature shows in either a row or a column
                counts = np.bincount(
                    np.concatenate(np.where(mask))
                )
                # Get the row/col with the most nans
                i = np.argmax(counts)
                # Keep track of the index and the feature name to remove
                idx_to_remove.append(i)
                feats_to_remove.append(ranks.columns[i])
                # Remove the nans in the corresponding row and column
                mask[i, :] = False
                mask[:, i] = False
            print(f'done: {(time.time() - t0) / 60:.2f} min')
            print('   Filtering correlation matrix ... ', end = "")
            t0 = time.time()
            # Now make the mask for the correlation matrix 
            mask_to_keep[idx_to_remove] = False
            
            corr = corr[mask_to_keep][:,mask_to_keep]
            print(f'done: {(time.time() - t0) / 60:.2f} min')
            if len(feats_to_remove) > 0:
                print(f'Removed the following {len(feats_to_remove)} of {total_nans} features with NaNs:')
                print(f'{feats_to_remove}')
            del upper_mask, mask, idx_to_remove
            gc.collect()
    t0 = time.time()
    # Make distance matrix and calculate linkage
    print("   Calculating Distance and linkage matrix ... ", end = "")
    dist = 1 - corr
    del corr
    gc.collect()
    dist_condensed = squareform(dist)
    Z = linkage(dist_condensed, method = 'average') 
    print(f"done: {(time.time() - t0) / 60:.2f} min")

    # Calculate best threshold for clustering:
    best_labels = []
    best_score = 0
    best_t = 0
    thresholds = []
    scores = []

    # For testing, use 100 log-uniform thresholds from [0.001, 0.1] and 100 bins from (.1, max(threshold)]
    thresholds_all = np.concat([np.logspace(-3, -1, 100), np.linspace(.1, np.max(Z[:,2]), 100)[1:]])

    def get_silhouette_score(Z, t, dist, max_clusters, batch_size = None, random_state = None):
        """
        Given linkage Z, threshold t, distance matrix dist, and a max number of clusters,
        return the cluster labels, silhouette score, and threshold
        """
        labels = fcluster(Z, t = t, criterion = 'distance')
        n_clusters = len(np.unique(labels))
        
        if n_clusters <= 1 or n_clusters >= max_clusters:
            return None
        
        if batch_size is not None:
            scores = []
            
            # Repeat sampling until standard error on the mean is < .01
            SEM = 1
            while SEM > .01 and len(scores) < 10:
                scores.append(
                    silhouette_score(
                        dist,
                        label = labels,
                        metric = 'precomputed',
                        sample_size = batch_size,
                        random_state = random_state + len(scores) if random_state is not None else random_state)
                )
                se = np.std(scores) / np.sqrt(len(scores))
                SEM = se / np.mean(scores)
            score = np.mean(scores)
        else:
            score = silhouette_score(dist, labels = labels, metric = 'precomputed', sample_size = batch_size, random_state = random_state)
        return (labels, score, t)

    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() - 1
    if n_jobs > 1:
        tasks = [delayed(
            get_silhouette_score)(
                Z, t, dist, len(dist), batch_size = silhouette_batch_size, random_state = random_state
            ) for t in thresholds_all
        ]
        
        with tqdm_joblib(tqdm(desc = "Calculating silhouette scores", total = len(tasks))) as pbar:
            results = Parallel(n_jobs = n_jobs)(tasks)
    else:
        results = [
            get_silhouette_score(
                Z, t, dist, len(dist), batch_size = silhouette_batch_size, random_state=random_state
            ) for t in tqdm(thresholds_all, desc = "Calculating silhouette scores")
        ]
        
    for res in results:
        if res is None:
            continue
        scores.append(res[1])
        thresholds.append(res[2])
        if res[1] > best_score:
            best_t = res[2]
            best_score = res[1]
            best_labels = res[0]

    print(f"   Clustering at threshold t = {best_t:.3f}")
    
    if plot_threshold:
        fig, ax = plt.subplots()
        ax.plot(thresholds, scores)
        ax.axvline(x = best_t, color = 'r', linestyle = '--', label = f'Best threshold = {best_t:.3f}')
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Silhouette score")
        ax.legend()
        plt.show()
        
    # Add labels to correlation dataframe
    clusters = pd.Series(best_labels, index = ranks.columns[mask_to_keep], name = 'cluster')
    # Add labels to the LLR df
    clusters_df = pd.merge(llr_df, clusters, left_index = True, right_index = True, how = 'inner')
    # Calculate simes p-value for each group
    simes_df = clusters_df.groupby('cluster')[pval_col].apply(get_simes_p)
    simes_df = simes_df.rename('p-simes').to_frame()
    simes_df['p-simes-adj'] = false_discovery_control(simes_df['p-simes'])
    # Add the simes df for future use
    clusters_df = pd.merge(clusters_df, simes_df, left_on = 'cluster', right_index = True, how = 'left')
    # Apply BH correction within each cluster
    clusters_df = clusters_df.groupby('cluster').apply(lambda g: apply_fdr(g, pval_col = pval_col))
    clusters_df.index = clusters_df.index.get_level_values(1)
    
    # Define the final fdr-q-value as the smallest alpha at which a feature would be significant
    clusters_df['fdr-q-val'] = 1.0
    for a in tqdm(np.linspace(1, 0, 1001), desc = "Calculating final FDR q-values"):
        R = len(simes_df[simes_df['p-simes-adj'] < a])
        m = len(simes_df)
        a_adj = a * R / m
        signif = (clusters_df['p-simes-adj'] < a) & (clusters_df['cluster-p-adj'] < a_adj)
        clusters_df.loc[signif, 'fdr-q-val'] = a
    # Significant if the fdr-q-value is below the preset alpha level
    clusters_df['signif'] = clusters_df['fdr-q-val'] < alpha
    
    print(f"Done with FDR corrections: {(time.time() - t_start) / 60:.2f} min")
    return clusters_df

def plot_posthoc(glms, pvals, feature, order = None, ylabel = 'Proportion', ax = None, title = "", hide_non_significant = True):
    """
    Plot the posthoc comparisons using means of the fits and pairwise comparisons
    
    Parameters:
    -----------
    glms: GLMCollection
        Fitted GLMCollection for all models
    pvals: pd.DataFrame
        posthoc dataframe (formatted as shown in tutorial notebook)
    feature: str
        Feature name for comparison
    order: List[str] | None
        Order for x-axis categories
    ylabel: str
        Y axis label
    title: str
        axis label
    """
    
    pvals_df = pvals[pvals['feature'] == feature].copy()
    pairs = [[g1, g2] for g1, g2 in zip(pvals_df['group1'].values, pvals_df['group2'].values)]
    ps = pvals_df['fdr-q-val'].values
    x_pos = {cat:i for i, cat in enumerate(glms.features[feature]['classification'].unique())}
    # If given a fixed order, make sure that classifications are present in the model
    if order is not None:
        if len(set(order).intersection(set(glms.features[feature]['classification'].unique()))) == 0:
            print(f"Given order has no values in the model's classification categories.\nPossible values are: {', '.join(self.features[model]['classification'].unique())}")
        else:
            x_pos = {cat: i for i, cat in enumerate(order)}

    # Make the dataframe
    means = []
    err_low = []
    err_high = []
    tmp = pd.DataFrame()
    missing = []
    for cat in x_pos.keys():
        if cat not in glms.features[feature]['classification'].unique():
            missing.append(cat)
            continue
        beta = glms.cond(model = feature, **{cat.split('__')[0]: cat.split('__')[1]})
        c = pd.Series(0, index = glms.results[feature].params.index)
        c.loc[beta.index] = beta
        t_res = glms.results[feature].t_test(c)
        means.append(glms.models[feature].link.inverse(t_res.effect.item()))
        errors = t_res.conf_int().squeeze()
        err_low.append(means[-1] - glms.models[feature].link.inverse(errors[0]))
        err_high.append(glms.models[feature].link.inverse(errors[1]) - means[-1])
        tmp = pd.concat([tmp, pd.DataFrame({'mean': [means[-1]], 'upper': [means[-1] + err_high[-1]], 'lower': [means[-1] - err_low[-1]]}, index = [cat])])

    if ax is None:
        fig, ax = plt.subplots()

    sns.stripplot(
        x = tmp.index,
        y = 'upper', 
        data = tmp,
        order = order,
        hue_order = order,
        ax = ax,
        facecolor = 'none',
        edgecolor = 'none',
    )

    for i, (cat, row) in enumerate(tmp.iterrows()):
        ax.errorbar(
            x_pos[cat],
            row['mean'],
            yerr = [[row['mean'] - row['lower']], [row['upper'] - row['mean']]],
            fmt = 'o',
            linestyle = 'none',
            c = f'C{x_pos[cat]}'
        )

    # Add missing columns
    # This is necessary for the annotation formatting to account for empty columns
    data = glms.features[feature].copy()
    for m in missing:
        data = pd.concat([data, pd.DataFrame({'rate': [np.nan], 'classification': [m]})], axis = 0, join = 'outer')
    annot = Annotator(ax, pairs, data = data, x = 'classification', y = 'rate', order = order)
    if sum(ps <= 0.05) == 0: # If no significant p-values, show all values
        hide_non_significant = False
    annot.configure(test = None, text_format = 'full', show_test_name = False, verbose = 0, hide_non_significant = hide_non_significant)
    annot.set_pvalues(ps)
    annot.annotate()

    ax.grid()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis = 'x', labelrotation = 45)
    return ax

def plot_posthoc_gene_analyzer(analyzer, pvals, feature, order = None, ylabel = 'Proportion', ax = None, title = "", hide_non_significant = True):
    """
    Plot the posthoc comparisons for the gene analyzer using means of the fits and pairwise comparisons
    
    Parameters:
    -----------
    analyzer: GeneAnalyzer
        Fitted GeneAnalyzer for all models
    pvals: pd.DataFrame
        posthoc dataframe (formatted as shown in tutorial notebook)
    feature: str
        Feature name for comparison
    order: List[str] | None
        Order for x-axis categories
    ylabel: str
        Y axis label
    title: str
        axis label
    """
    
    # Get the data for the given feature and get variables for the annotator
    data = analyzer.get_feature_dataframe(feature)
    pvals_df = pvals[pvals['feature'] == feature].copy()
    pairs = [[g1, g2] for g1, g2 in zip(pvals_df['group1'].values, pvals_df['group2'].values)]
    ps = pvals_df['fdr-q-val'].values
    # If given a fixed order, make sure that classifications are present in the model
    if order is not None:
        if len(set(order).intersection(set(data['classification'].unique()))) == 0:
            print(f"Given order has no values in the model's classification categories.\nPossible values are: {', '.join(data['classification'].unique())}")
    if ax is None:
        _, ax = plt.subplots()
        
    # Plot the feature
    analyzer.plot_feature(feature, order = order, ax = ax)
    
    # Add missing columns
    # This is necessary for the annotation formatting to account for empty columns
    missing = set(order).difference(set(data['classification'].unique()))
    for m in missing:
        data = pd.concat([data, pd.DataFrame({'rate': [np.nan], 'classification': [m]})], axis = 0, join = 'outer')
    annot = Annotator(ax, pairs, data = data, x = 'classification', y = feature.split('___')[0], order = order)
    if sum(ps < 0.05) == 0: # If no significant p-values, show all values
        hide_non_significant = False
    annot.configure(test = None, text_format = 'full', show_test_name = False, verbose = 0, hide_non_significant = hide_non_significant)
    annot.set_pvalues(ps)
    annot.annotate()

    ax.tick_params(axis = 'x', labelrotation = 45)
    return ax

def plot_test_stat(perm_df, stat_df, model, stat_col = 'stat', ax = None, bins = 'auto'):
    """
    Function to plot a histogram of the permutation dataframe test statistics
    Draws a vertical line at the location of the observed log likelihood
    
    Parameters:
    ------------
    perm_df: pd.DataFrame
        Dataframe containing permutation results for all models
    stat_df: pd.DataFrame
        Dataframe with observed test statistics
    model: str
        Name of feature/model to plot
    stat_col: str
        Name of test statistic column in dataframes
    ax: matplotlib.pyplot.axis | None
        matplotlib axis for plotting
    """
    if ax is None:
        fig, ax = plt.subplots()
    sns.histplot(
        x = stat_col,
        data = perm_df[perm_df.index == model],
        stat = 'probability',
        bins = bins,
        label = 'Permutation Tests'
    )
    ax.axvline(stat_df.loc[model][stat_col], color = 'red', linestyle = '--', label = 'Obs')
    ax.legend()
    ax.set_xlabel("Test Stat")
    ax.set_ylabel("a.u.")
    ax.set_title("Log Likelihood Ratios")
    return ax