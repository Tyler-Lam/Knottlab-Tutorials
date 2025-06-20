# Notebook to merge outputs from foundation models, converting from .h5 to .h5ad files, then apply the output to the xenium data
import h5py
from pathlib import Path
import os
import h5py
import anndata as ad
import numpy as np
import pandas as pd
import glob
import tqdm
import json
from shapely.geometry import Polygon, box
from collections import defaultdict
import scanpy as sc
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import time
import sys
import gc

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--tile_len", default = 224, type = int) # Length of tiles in pixels
parser.add_argument('--patch_len', default = 14, type = int) # Length of patches in pixels
parser.add_argument("--overlap", default = 0, type = int)    # Amount of overlap between patches (in pixels)
parser.add_argument("--model", default = 'virchow2', type = str) # Model used to run h&e feature extraction
parser.add_argument('--slide', default = 'ag_hpv_01', type = str) # Slide number (in h&e naming) to merge features on

args = vars(parser.parse_args())
overlap = args['overlap']
tile_len = args['tile_len']
pixels_per_patch = args['patch_len']
model = args['model']
slide = args['slide']

import warnings
import anndata

# Filter out the "switching to string index" warning that messes with tqdm
warnings.filterwarnings('ignore', category=anndata.ImplicitModificationWarning) 

# Given coords of patch corners and polygon defining tissue segmentation
# calculate the fraction of each patch that is within the segmentation
def get_frac_background(x0, y0, x1, y1, segmentation):
    patch = box(x0, y0, x1, y1)
    intersect = patch.intersection(segmentation)
    return (intersect.area / patch.area)

# Flatten features and coordinates when using small patch embeddings
def unpack_h5_file(tile_coords, tile_features, patch_coords, patch_features):
    parent_coords = np.repeat(tile_coords, [patch_features.shape[1] for _ in range(patch_features.shape[0])], axis = 0)
    patch_coords = np.vstack([patch_coords] * patch_features.shape[0])
    coords_out = patch_coords + parent_coords
    
    # Weight based on contribution of each patch to the aggregate of overlapping tiles
    # Equivalently, the weight is equal to the area between the patch and the closest tile corner
    # This only works for 50% overlap, need to generalize for different fractions
    if tile_len / overlap == 2:
        patch_midpoint = coords_out + [pixels_per_patch/2, pixels_per_patch/2]
        delta_xy1 = patch_midpoint - parent_coords
        delta_xy2 = tile_len + parent_coords - patch_midpoint
        
        delta_xy = np.minimum(delta_xy1, delta_xy2)
        
        weights = delta_xy[:,0] * delta_xy[:,1]
    else:
        weights = np.ones(len(coords_out))
    
    cls = np.repeat(tile_features[:,:tile_features.shape[1]//2], [patch_features.shape[1] for _ in range(tile_features.shape[0])], axis = 0)
    features_out = np.vstack(patch_features) 
    features_out = np.concat((features_out, cls), axis = 1)
    return (coords_out, features_out, parent_coords, weights)


# Convert h5 to h5ad, calculate frac background for each patch, and merge
def create_anndata_from_h5_files_with_segmentation(directory_path):
    """
    Creates an AnnData object from all H5 files in a directory and return AnnData.
    
    Args:
        directory_path (str): The path to the directory containing the H5 files.
    """
    # Find all H5 files in the directory for a given slide
    h5_files = glob.glob(os.path.join(directory_path, f"{slide}___*.h5"))
    
    if not h5_files:
        print(f"No H5 files found in '{directory_path}'")
        return
    
    # List to store individual AnnData objects for concatenating
    anndata_objects = []
        
    # Process each H5 file
    for file_path in (pbar := tqdm.tqdm(h5_files)):
        file_name = os.path.basename(file_path)  # Get the filename without path
        try:
            with h5py.File(file_path, 'r') as file:
                
                pbar.set_description("Getting segmentation from geojson")
                # Read in segmentation from the geojson and make the polygons:
                json_file = open(f'/common/lamt2/HPV/data/foundation_models/segmentation/{file_path.split('/')[-1].replace('.h5', '')}.geojson')
                data = json.load(json_file)
                segmentation = defaultdict(Polygon) # Currently geojsons only have 2 classes: tissue and folds
                for feature in data['features']:
                    coords = feature['geometry']['coordinates']
                    for coord in coords:
                        feat_shape = Polygon(coord)
                        segmentation[feature['properties']['classification']['name']] = segmentation[feature['properties']['classification']['name']].union(feat_shape)
                        
                # Read the coords and features dataset
                # If we have the patch information, use the patches as the rows in the anndata
                if 'coords' in file and 'features' in file and 'internal_patch_coords' in file and 'patch_embeddings' in file:
                    # Properties of full tiles
                    tile_coords = file['coords'][()]
                    tile_features = file['features'][()]

                    # Properties of patches
                    patch_features= file['patch_embeddings'][()]
                    patch_coords = file['internal_patch_coords'][()]
                    
                    num_records = patch_features.shape[0] * patch_features.shape[1]
                    
                    pbar.set_description("Unpacking h5 file")
                    coords, features, parent_coords, weights = unpack_h5_file(tile_coords, tile_features, patch_coords, patch_features)
                    patch_idx = [f'patch_{i}_tile_{j}_{file_name.replace('.h5', '')}' for j in range(patch_features.shape[0]) for i in range(patch_features.shape[1]) ]
                    #patch_idx = np.array([[f'patch_{i}_tile_{j}_{file_name.replace('.h5', '')}' for i in range(patch_features.shape[1])] for j in range(patch_features.shape[0])]).reshape(features.shape[0], -1)
                    # Create an AnnData object
                    adata = ad.AnnData(
                        X=features,
                        obs = pd.DataFrame({'x0': coords[:,0], 'y0': coords[:,1], 'parent_x0': parent_coords[:,0], 'parent_y0': parent_coords[:,1], 'weight_raw': weights}, index = patch_idx)
                    )
                    
                    del features, coords, parent_coords, patch_idx
                    gc.collect()
                    
                    adata.obs['batch'] = file_name.split('/')[-1].split('.')[0]
                    adata.obs['core'] = adata.obs['batch'].apply(lambda x: x.split('___')[1])
                    adata.obs['slide'] = adata.obs['batch'].apply(lambda x: x.split('___')[0])
                    pbar.set_description("Calculating fractions from segmentation")
                    for annot in segmentation:
                        #adata.obs[f'frac_{annot}'] = adata.obs.apply(lambda x: get_frac_background(x['x0'], x['y0'], x['x0'] + pixels_per_patch, x['y0'] + pixels_per_patch, segmentation[annot]), axis = 1)
                        #adata.obs[f'parent_frac_{annot}'] = adata.obs.apply(lambda x: get_frac_background(x['parent_x0'], x['parent_y0'], x['parent_x0'] + tile_len, x['parent_y0'] + tile_len, segmentation[annot]), axis = 1)

                        #adata.obs[f'frac_{annot}'] = np.vectorize(get_frac_background)(adata.obs['x0'], adata.obs['y0'], adata.obs['x0'] + pixels_per_patch, adata.obs['y0'] + pixels_per_patch, segmentation[annot])
                        adata.obs[f'parent_frac_{annot}'] = np.vectorize(get_frac_background)(adata.obs['parent_x0'], adata.obs['parent_y0'], adata.obs['parent_x0'] + tile_len, adata.obs['parent_y0'] + tile_len, segmentation[annot])
                    
                    anndata_objects.append(adata)
                    del adata
                
                # If no patches, only use the tiles
                elif 'coords' in file and 'features' in file:
                    coords = file['coords'][()]
                    features = file['features'][()]
                    xs = [x[0] for x in coords]
                    ys = [x[1] for x in coords]
                    # Create an index for this file's data
                    num_records = coords.shape[0]
                    indexes = [f"tile_{i}_{file_name.replace('.h5', '')}" for i in range(num_records)]
                    
                    # Create an AnnData object
                    adata = ad.AnnData(
                        X=features,
                        obs=pd.DataFrame({'x0': xs, 'y0': ys}, index=indexes),
                    )
                    adata.obs['batch'] = file_name.split('/')[-1].split('.')[0]
                    adata.obs['core'] = adata.obs['batch'].apply(lambda x: x.split('___')[1])
                    adata.obs['slide'] = adata.obs['batch'].apply(lambda x: x.split('___')[0])
                    for annot in segmentation:
                        adata.obs[f'frac_{annot}'] = adata.obs.apply(lambda x: get_frac_background(x['x0'], x['y0'], x['x0'] + tile_len, x['y0'] + tile_len, segmentation[annot]), axis = 1)

                    anndata_objects.append(adata)

                    del adata
                
                else:
                    print(f"Warning: '{file_name}' doesn't contain both 'coords' and 'features' datasets")
        except Exception as e:
            print(f"Error processing '{file_name}'")
            import traceback
            err = traceback.format_exc()
            print(err)
    print('Finished converting h5 files')
    if not anndata_objects:
        print("No valid data found in any of the H5 files")
        return
    try:
        # Combine all AnnData objects
        combined_adata = ad.concat(
            anndata_objects,
            join='outer',  # Use outer join to include all variables
            index_unique=None  # We've already created unique indices
        )
    except Exception as e:
        print("Error combining anndata objects: {e}")

    print(f"Total records: {combined_adata.shape[0]}")
    return combined_adata

print('Translating h5 files to h5ad files')
adata = create_anndata_from_h5_files_with_segmentation(directory_path=f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/features_{model}/')

# Replace nan with 0
for col in adata.obs.columns:
    adata.obs[col] = adata.obs[col].replace(np.nan, 0)

# Filter based on virchow2 training criteria
if 'parent_frac_fold' in adata.obs.columns:
    adata = adata[(adata.obs['parent_frac_tissue'] > 0.65) & (adata.obs['parent_frac_fold'] < 0.35)]
else:
    adata = adata[(adata.obs['parent_frac_tissue'] > 0.65)]

# Normalize patch weights
adata.obs['weight'] = adata.obs.groupby(['x0', 'y0', 'core'])['weight_raw'].transform(lambda x: x/x.sum())

# If overlap, re-segment patches by averaging overlapping tiles
if overlap != 0:
    def resegment_h5ad(adata):
        if 'parent_x0' in adata.obs.columns and 'parent_y0' in adata.obs.columns:
            X_weighted = adata.X * np.vstack(adata.obs['weight'])
            adata.X = X_weighted
            adata_agg = sc.get.aggregate(adata, by = ['x0', 'y0', 'core'], func = 'sum')
            adata_agg.X = adata_agg.layers['sum']
            adata_agg.obs['slide'] = slide
            adata_agg.obs.reset_index()
            patch_idx = adata_agg.obs.groupby('core', observed = False).cumcount()
            adata_agg.obs['patch_idx'] = patch_idx
            adata_agg.obs['batch'] = adata_agg.obs['slide'].astype(str) + '___' + adata_agg.obs['core'].astype(str) + '___' + adata_agg.obs['patch_idx'].astype(str)
            adata_agg.obs.set_index('batch', inplace = True)
            return adata_agg
        
        else:
            adata_per_batch = []
            for batch in (pbar := tqdm.tqdm(adata.obs['batch'].unique(), desc = "Resegmenting anndata patches")):
                try:
                # If we only have tile information, re-segment using geometry of the tiles
                    xys = adata[adata.obs['batch'] == batch].obs[['x0', 'y0']]
                    
                    # Construct line segments for all tiles:
                    
                    # First create a list of all tiles using shapely's box class
                    patches = []
                    for idx, row in xys.iterrows():
                        patch = box(row.x0, row.y0, row.x0 + tile_len, row.y0 + tile_len)
                        patches.append(patch)
                    boxes = [LineString(list(pol.exterior.coords)) for pol in patches]
                    # Get the line segments from the union of all tiles
                    union = unary_union(boxes)
                    # Divide the line segments into polygons again
                    results = [geom for geom in polygonize(union)]
                    
                    # For each resegmented tile, use the midpoint to get all original tiles that overlap and average the features
                    x0 = []
                    x1 = []
                    y0 = []
                    y1 = []
                    features = []
                    patch_idx = []
                    i = 0
                    for res in tqdm.tqdm(results, leave = False):
                        xi = int(min(res.exterior.xy[0]))
                        xf = int(max(res.exterior.xy[0]))
                        yi = int(min(res.exterior.xy[1]))
                        yf = int(max(res.exterior.xy[1]))
                        
                        # Use the midpoint of the patch to get the patches that overlap the sub patch
                        x = (xi + xf) / 2
                        y = (yi + yf) / 2
                        adata_overlapping = adata[
                            (adata.obs['x0'] <= x) &
                            (adata.obs['y0'] <= y) &
                            (adata.obs['x0'] + tile_len > x) &
                            (adata.obs['y0'] + tile_len > y) &
                            (adata.obs['batch'] == batch)
                        ]
                        
                        # Occasionally shapely finds a "patch" that wasn't filled (ie, hole surrounded by 4 real patches). Skip those
                        if len(adata_overlapping) == 0:
                            continue
                        
                        x0.append(xi)
                        x1.append(xf)
                        y0.append(yi)
                        y1.append(yf)
                        patch_idx.append(f'patch_{i}_{batch}')
                        features.append(np.mean(adata_overlapping.X, axis = 0))

                        i+=1
                    tmp = ad.AnnData(
                        X = np.array(features),
                        obs = pd.DataFrame({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}, index = patch_idx)

                    )
                    tmp.obs['batch'] = batch
                    tmp.obs['slide'] = batch.split('___')[0]
                    tmp.obs['core'] = batch.split('___')[-1]

                    adata_per_batch.append(tmp)
                    
                except Exception as e:
                    print(f"Error processing '{batch}'")
                    import traceback
                    err = traceback.format_exc()
                    print(err)                
            try:
                out = ad.concat(
                    adata_per_batch,
                    join = 'outer',
                )
            except Exception as e:
                print (f'Error concatenating anndata: {e}')
            return out
    t0 = time.time()
    print("Resegmenting adata from overlapping tiles")
    adata = resegment_h5ad(adata)
    print(f"...done: {(time.time() - t0)/60:.2f} min")
    
# Associate xenium slide name to h&e slide name
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
print("Reading xenium anndata .... ", end = '')
adata_xenium = sc.read_h5ad('/common/lamt2/HPV/data/xenium/adata/adata_qc.h5ad')

# Cropping coordinate and pixel conversions
df_idx = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col = 0)
sf = 0.525548 # micron per pixel for foundation model coords (Assuming 20x magnification for h&e images)
level_factor = 3.4 # microns per pixel for cropping coords (Assuming coords calculated on level 4 dapi)

# Filter to only include the current slide
adata_xenium.obs['slide_idx'] = adata_xenium.obs['slide'].map(anno_dict)
ad_slide = adata_xenium[adata_xenium.obs['slide_idx'] == slide]
df_idx['slide_idx'] = df_idx['slide'].map(anno_dict)
df_idx = df_idx[df_idx['slide_idx'] == slide]

print(f"done: {(time.time() - t0)/60:.2f} min")

# Function to find foundation model patch based on x-y cell coordinate, and return the foundation model representation
# Try using numba njit and prange to make looping over arrays faster
# Takes all inputs as arrays from the pandas dataframe
from numba import njit, prange, get_num_threads
@njit(parallel = True)
def get_X_foundation_model(
    xs, ys, cores, # Taken from the xenium data
    crop_coords, crop_core, # Taken from the cropping coordinate dataframe
    patch_coords, patch_cores, X # Taken from the merged foundation model outputs
    ):
    
    res = np.empty((len(xs), X.shape[1]), dtype = np.float32) #Allocate output array
    
    # Loop over cells
    for i in prange(len(xs)):
        x_centroid = xs[i]
        y_centroid = ys[i]
        core = cores[i]
        
        # Get cropping coords
        match = crop_coords[(crop_core == core)]
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
                (patch_cores == core)]
            # If no core found, return nans
            if (len(rep) == 0):
                res[i] = np.full(X.shape[1], np.nan)
            # Otherwise return foundation model features
            else:
                res[i] = rep[0].astype(np.float32)
    return res

# Get foundation model features for each cell
t0 = time.time()
print(f"Transferring foundation model to cells (using {get_num_threads()} threads).... ", end = '')

X_model = get_X_foundation_model(
    ad_slide.obs['x_centroid'].to_numpy(dtype=np.int64), 
    ad_slide.obs['y_centroid'].to_numpy(dtype=np.int64), 
    ad_slide.obs['core_idx'].to_numpy(dtype=np.int64), 
    df_idx[['x0', 'y0']].to_numpy(dtype=np.int64), 
    df_idx['core'].to_numpy(dtype=np.int64), 
    adata.obs[['x0', 'y0']].to_numpy(dtype=np.int64), 
    adata.obs['core'].to_numpy(dtype=np.int64), 
    adata.X)

print(f'done: {(time.time() - t0)/60:.2f} min')
t0 = time.time()
print("Saving feature matrix .... ", end = '')
# Save just the foundation model feature matrix
np.save(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/X_foundation_model_{slide}_weighted.npy', X_model)
print(f'done: {(time.time() - t0)/60:.2f} min')
print('---done---')