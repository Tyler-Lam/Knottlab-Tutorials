import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import math
import time
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from scipy.stats import beta, mannwhitneyu
from scipy.stats import false_discovery_control
from patsy import dmatrices
import statsmodels.api as sm
from formulaic_contrasts import FormulaicContrasts
from joblib import Parallel, delayed, parallel_backend
from itertools import combinations, product
import warnings
from collections import defaultdict
from pathlib import Path
import os
import re
import multiprocessing
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib import colors, gridspec

from statsmodels.base.model import GenericLikelihoodModel, GenericLikelihoodModelResults
from statsmodels.othermod.betareg import BetaModel, BetaResults, BetaResultsWrapper
from statsmodels.genmod.generalized_linear_model import GLMResults
from scipy.special import betaln
from scipy.special import gammaln
from scipy.stats import lognorm
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

            
# Stackoverflow solution to make tqdm work with joblib.Parallel
# https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
import contextlib
import joblib
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
        

# Log(n choose k) for stability in log-likelihood functions
def log_comb(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def get_celltype_annot_region_feature_type(col):
    """
    Given a column name, return the cell type, annotation region, and feature type. This assumes columns formatted according to the tutorial
    Feature types that this can return are:
    * {cell_category}_proportion
    * {cell_category}_density
    * {child_category}_per_{parent_celltype}
    * {cell_category}_spatial_correlation_central_cell_{central_celltype}
    * {cell_category}_covering_fraction_{celltype_1}
    where {cell_category} is primary_celltype, secondary_celltype, etc.
    """
    
    region = col.split('annot_region_')[1][0] if 'annot_region' in col else '0'
    celltype = col.split('___')[-1]
    annot_dict = {'2': 'Stroma', '3': 'Epithelium', '0': 'All Tissue'} # For our project, we only have two values for our annotation masks
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
    * df: Stats dataframe returned from GLMCollection
    *
    * x_col, y_col: dataframe col names to place the row/column for each feature
    * hue_col: Dataframe column to color each point
    * pval_col: Dataframe column for pvalue (size + significance)
    * pval_threshold: Minimum p-value to indicate feature is significant
    * vmax: Maximum abs(hue_col) for color normalization
    * xlabel, ylabel: Label for x/y-axis (defaults to x_col/y_col)
    * cbar_label: Label for the colorbar
    * test_group, ref_group: Labels for colorbar to indicate enrichment in test vs ref class
    * title: Title for the plot
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
            


def get_progression_df(stat_res, ref_group):
    """
    Given a statistics result from the GLMCollection, aggregate the dataframes to give feature effects relative to the reference group
    
    Parameters:
    ------------
    * stat_res: Output from GLMCollection.run_stats() (or run_stats_with_permutations())
    * ref_group: Dictionary defining the reference group predictors
    """
    res_df = pd.DataFrame()
    
    for key, res in stat_res.items():
        if key == 'llr':
            continue
        tmp_df = res['df'].copy()
        if res['ref_group'] == ref_group:
            for col, val in res['test_group'].items():
                tmp_df[col] = val
        
        elif res['test_group'] == ref_group:
            for col, val in res['ref_group'].items():
                tmp_df[col] = val
                
            tmp_df['effect'] = -1 * tmp_df['effect']
        
        else:
            continue

        res_df = pd.concat([res_df, tmp_df], axis = 0, join = 'outer')
    
    res_df = res_df.reset_index()
    res_df = res_df.rename(columns = {'index': 'feature'})
    return res_df

def plot_progression_heatmap(
    stat_df, llr_df = None, index = ['Celltype', 'Region'], columns = 'Diagnosis', values = 'effect',
    ncols = 1, vmax = None, title = "", xlabel = None, cbar_label = 'log2FoldChange', row_cluster = True,
    pval_col = 'p-adj', cbar_width = 0.03, cbar_height = 0.45, order = None, figsize = (10,10), show = True, save_dir = None):
    """
    Make the heatmap for a feature across several values of a predictor. 
    Makes a pivot table and pass to seaborn clustermap and does additional formatting
    I don't know what seaborn does but making adjustments (like placing legends and padding) are very hard to do from the clustermap
    
    Parameters:
    -----------
    * stat_df: Dataframe obtained from get_progression_df
    * llr_df: Dataframe of Log likelihood ratio test results (used to label features using FDR q-value)
    * index: Columns used to index the pivot table
    * columns: Col name to use a columns for the pivot table
    * values: Values for the pivot table
    * ncols: Number of columns to use for celltype legend. Values >1 only really works for niches
    * vmax: Maximum value for heatmap normalization
    * title: Title for the plot
    * xlabel: Label for the x-axis (defaults to "columns" parameter)
    * cbar_label: Label for the colorbar
    * row_cluster: Cluster the rows and plot with a dendrogram. Otherwise order the rows based on the index
    * pval_col: Column name for feature p-value labels on clustermap
    * cbar_width: width of colorbar
    * cbar_height: maximum height of colorbar
    * order: Left to right order for the columns
    """

    # Make the pivot table for the clustermap
    pivot = stat_df.pivot(index = index, columns = columns, values = values)
    mask = pivot.sum(axis = 1) == 0 # Remove columns without comparisons
    pivot = pivot[~mask]
    pivot = pivot.sort_index(level = range(len(index)))
    
    if order is None:
        order = pivot.columns.unique()
        
    if len(pivot) == 0:
        return
    if len(pivot) == 1:
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
        for label, tick in zip(ylabels, yticks):
            tick.set_text(f'p={label:.3f}')
            if label < 0.05:
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
