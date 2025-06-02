# Script to assign foundation model patch features to individual cells in xenium data
import scanpy as sc
import pandas as pd
import numpy as np
import tqdm
import os
import time

# Arguments for foundation model
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--tile_len", default = 224, type = int)
parser.add_argument('--patch_len', default = 14, type = int)
parser.add_argument("--overlap", default = 0, type = int)
parser.add_argument("--model", default = 'virchow2', type = str)

args = vars(parser.parse_args())

overlap = args['overlap']
tile_len = args['tile_len']
pixels_per_patch = args['patch_len']
model = args['model']

# Associate xenium slide name to h&e slide name (needed to assign consistent slide index as int)
anno_dict = {
    '20250213__202616__X206_02132025_ANOGENTMA_1_2/output-XETG00206__0060075__Region_1__20250213__202651': 'ag_hpv_01',
    '20250213__202616__X206_02132025_ANOGENTMA_1_2/output-XETG00206__0060077__Region_1__20250213__202651': 'ag_hpv_02',
    '20250224__233848__X206_2242025_ANOGENTMA_03_04/output-XETG00206__0060354__Region_1__20250224__233922': 'ag_hpv_04',
    '20250224__233848__X206_2242025_ANOGENTMA_03_04/output-XETG00206__0060367__Region_1__20250224__233922': 'ag_hpv_03',
    '20250304__005745__X403_03032025_ANOGENTMA_05_06/output-XETG00403__0059911__Region_1__20250304__005817': 'ag_hpv_06',
    '20250304__005745__X403_03032025_ANOGENTMA_05_06/output-XETG00403__0060395__Region_1__20250304__005817': 'ag_hpv_05',
    '20250305__223640__X206_03052025_HPVTMA_01_02/output-XETG00206__0060364__Region_1__20250305__223715': 'ag_hpv_08',
    '20250305__223640__X206_03052025_HPVTMA_01_02/output-XETG00206__0060366__Region_1__20250305__223715': 'ag_hpv_07',
    '20250312__003942__X206_03112025_HPVTMA_03_04/output-XETG00206__0060488__Region_1__20250312__004017': 'ag_hpv_09',
    '20250312__003942__X206_03112025_HPVTMA_03_04/output-XETG00206__0060493__Region_1__20250312__004017': 'ag_hpv_10'
}
t0 = time.time()
print("Reading anndata", end = '')
adata = sc.read_h5ad('/common/lamt2/HPV/data/xenium/adata/adata_qc.h5ad')

df_idx = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col = 0)
sf = 0.525548 # micron per pixel for foundation model coords (Assuming 20x magnification for h&e images)
level_factor = 3.4 # microns per pixel for cropping coords (Assuming coords calculated on level 4 dapi)

# This is so we can have a consistent slide index (integer needed for numba integration later)
def convert_slide_to_idx(slide):
    if slide in anno_dict:
        he_idx = anno_dict[slide]
    else:
        he_idx = slide
    return int(he_idx.split('_')[-1])

adata.obs['slide_idx'] = np.vectorize(convert_slide_to_idx)(adata.obs['slide'])
df_idx['slide_idx'] = np.vectorize(convert_slide_to_idx)(df_idx['slide'])

print(f" ... done: {(time.time() - t0)/60:.2f} min")

# Function to find foundation model patch based on x-y cell coordinate, and return the foundation model representation
# Try using numba njit and prange to make looping over arrays faster
# Takes all inputs as arrays from the pandas dataframe
from numba import njit, prange
@njit(parallel = True)
def get_X_foundation_model(
    xs, ys, slides, cores, # Taken from the xenium data
    crop_coords, crop_slide, crop_core, # Taken from the cropping coordinate dataframe
    patch_coords, patch_slides, patch_cores, X # Taken from the merged foundation model outputs
    ):
    
    res = np.empty((len(xs), X.shape[1]), dtype = np.float32) #Allocate output array
    
    # Loop over cells
    for i in prange(len(xs)):
        x_centroid = xs[i]
        y_centroid = ys[i]
        slide = slides[i]
        core = cores[i]
        
        # Get cropping coords
        match = crop_coords[(crop_slide == slide) & (crop_core == core)]
        # If no match: return empty array of nans
        if len(match) == 0:
            res[i] = np.full(X.shape[1], np.nan)
        else:
            # Get cell coordinates in pixel coordinates
            x = (x_centroid - match[0][0] * level_factor) / sf
            y = (y_centroid - match[0][1] * level_factor) / sf
            # Get foundation model of patch containing cell coordinations
            rep = X[
                (patch_coords[:,0] <= x) &
                (patch_coords[:,0] + 14 > x) &
                (patch_coords[:,1] <= y) &
                (patch_coords[:,1] + 14 > y) &
                (patch_slides == slide) &
                (patch_cores == core)]
            # If no core found, return nans
            if (len(rep) == 0):
                res[i] = np.full(X.shape[1], np.nan)
            # Otherwise return foundation model features
            else:
                res[i] = rep[0].astype(np.float32)
    return res

# Assign foundation model features to empty numpy array
#adata.obsm['foundation_model'] = np.empty((len(adata), 2560), dtype = np.float32)
t0 = time.time()

# Do for each slide separately (since files were too large to combine)
print("Looping over slides to fill rep matrix")
for idx in tqdm.tqdm(adata.obs['slide_idx'].unique()):
    f_name = f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/ag_hpv_{str(idx).zfill(2)}_combined.h5ad'
    print(f"Running on {f_name}")
    if not os.path.exists(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/ag_hpv_{str(idx).zfill(2)}_combined.h5ad'):
        print("   File does not exist, skipping")
        continue
    
    # Read in foundation model output
    ad_model = sc.read_h5ad(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/ag_hpv_{str(idx).zfill(2)}_combined.h5ad')
    ad_model.obs['slide_idx'] = np.vectorize(convert_slide_to_idx)(ad_model.obs['slide'])
    
    # Get cells for the given slide
    slide = adata[adata.obs['slide_idx'] == idx].obs
    
    # Get foundation model features for each cell
    X_model = get_X_foundation_model(
        slide['x_centroid'].values, 
        slide['y_centroid'].values, 
        slide['slide_idx'].values,
        slide['core_idx'].values, 
        df_idx[['x0', 'y0']].values, 
        df_idx['slide_idx'].values,
        df_idx['core'].values, 
        ad_model.obs[['x0', 'y0']].values, 
        ad_model.obs['slide_idx'].values, 
        ad_model.obs['core'].astype(np.int64).values, 
        ad_model.X)
    
    #np.save(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/X_foundation_model_ag_hpv_{str(idx).zfill(2)}.npy', X_model)
    # Assign features to each cell from original xenium anndata
    adata.obsm['foundation_model'][adata.obs['slide_idx'] == idx] = X_model
print(f'...done: {(time.time() - t0)/60:2f} min')

# Save just the foundation model feature matrix (todo: better compression)
np.save(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/X_foundation_model.npy', adata.obsm['foundation_model'])
print("Wrote feature matrix ... done")