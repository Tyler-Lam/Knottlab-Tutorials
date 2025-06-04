# Notebook to merge outputs from foundation models, converting from .h5 to .h5ad files
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

    cls = np.repeat(tile_features[:,:tile_features.shape[1]//2], [patch_features.shape[1] for _ in range(tile_features.shape[0])], axis = 0)
    features_out = np.vstack(patch_features) #patch_features.reshape(patch_features.shape[0] * patch_features.shape[1], -1)
    features_out = np.concat((features_out, cls), axis = 1)
    return (coords_out, features_out, parent_coords)

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
    for file_path in tqdm.tqdm(h5_files):
        file_name = os.path.basename(file_path)  # Get the filename without path
        try:
            with h5py.File(file_path, 'r') as file:
                
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
                    coords, features, parent_coords = unpack_h5_file(tile_coords, tile_features, patch_coords, patch_features)

                    patch_idx = np.array([[f'patch_{i}_tile_{j}_{file_name.replace('.h5', '')}' for i in range(patch_features.shape[1])] for j in range(patch_features.shape[0])]).reshape(features.shape[0], -1)
                    # Create an AnnData object
                    adata = ad.AnnData(
                        X=features,
                        obs = pd.DataFrame({'x0': coords[:,0], 'y0': coords[:,1], 'parent_x0': parent_coords[:,0], 'parent_y0': parent_coords[:,1]}, index = patch_idx)
                    )
                    
                    del features, coords, parent_coords#, patch_idx
                    gc.collect()
                    adata.obs['batch'] = file_name.split('/')[-1].split('.')[0]
                    adata.obs['core'] = adata.obs['batch'].apply(lambda x: x.split('___')[1])
                    adata.obs['slide'] = adata.obs['batch'].apply(lambda x: x.split('___')[0])
                    
                    for annot in segmentation:
                        #adata.obs[f'frac_{annot}'] = adata.obs.apply(lambda x: get_frac_background(x['x0'], x['y0'], x['x0'] + pixels_per_patch, x['y0'] + pixels_per_patch, segmentation[annot]), axis = 1)
                        #adata.obs[f'parent_frac_{annot}'] = adata.obs.apply(lambda x: get_frac_background(x['parent_x0'], x['parent_y0'], x['parent_x0'] + tile_len, x['parent_y0'] + tile_len, segmentation[annot]), axis = 1)

                        adata.obs[f'frac_{annot}'] = np.vectorize(get_frac_background)(adata.obs['x0'], adata.obs['y0'], adata.obs['x0'] + pixels_per_patch, adata.obs['y0'] + pixels_per_patch, segmentation[annot])
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
print('...done')

# Replace nan with 0
for col in adata.obs.columns:
    adata.obs[col] = adata.obs[col].replace(np.nan, 0)

# Filter based on virchow2 training criteria
#adata = adata[(adata.obs['parent_frac_tissue'] > 0.65) & (adata.obs['parent_frac_fold'] < 0.35)]

# If no overlap, save anndata
if overlap == 0:
    adata.write(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/{slide}_combined.h5ad')
    print('---done---')

# If overlap, re-segment patches by averaging overlapping tiles
else:
    def resegment_h5ad(adata):
        adata_per_batch = []
        for batch in tqdm.tqdm(adata.obs['batch'].unique()):
            try:
                # Read in segmentation from the geojson and make the polygon:
                json_file = open(f'/common/lamt2/HPV/data/foundation_models/segmentation/{batch}.geojson')
                data = json.load(json_file)
                segmentation = defaultdict(Polygon)
                for feature in data['features']:
                    coords = feature['geometry']['coordinates']
                    for coord in coords:
                        feat_shape = Polygon(coord)
                        segmentation[feature['properties']['classification']['name']] = segmentation[feature['properties']['classification']['name']].union(feat_shape)
                        
                # If we have patch information, use patches
                # This assumes that overlap is an integer multiple of the patch length, otherwise this doesn't work
                if 'parent_x0' in adata.obs.columns and 'parent_y0' in adata.obs.columns:
                    
                    # Use scanpy to aggregate patches with the same coordinates
                    adata_agg = sc.get.aggregate(adata[adata.obs['batch'] == batch], by = ['x0', 'y0'], func = 'mean')
                    adata_agg.X = adata_agg.layers['mean']
                    idx = [f'patch_{i}_{batch}' for i in range(len(adata_agg))]
                    adata_agg.obs['patch'] = idx
                    adata_agg.obs.set_index('patch')
                    adata_agg.obs['batch'] = batch
                    adata_agg.obs['slide'] = batch.split('___')[0]
                    adata_agg.obs['core'] = batch.split('___')[-1]

                    for annot in segmentation:
                        #adata_agg.obs[f'frac_{annot}'] = adata_agg.obs.apply(lambda x: get_frac_background(x['x0'].astype(int), x['y0'].astype(int), x['x0'].astype(int) + pixels_per_patch, x['y0'].astype(int) + pixels_per_patch, segmentation[annot]), axis = 1)
                        adata_agg.obs[f'frac_{annot}'] = np.vectorize(get_frac_background)(adata_agg.obs['x0'].astype(int), adata_agg.obs['y0'].astype(int), adata_agg.astype(int).obs['x0'] + pixels_per_patch, adata_agg.obs['y0'].astype(int) + pixels_per_patch, segmentation[annot])
                    adata_per_batch.append(adata_agg)

                # If we only have tile information, re-segment using geometry of the tiles
                else:
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
                    for annot in segmentation:
                        #tmp.obs[f'frac_{annot}'] = np.vectorize(get_frac_background)(tmp.obs['x0'], tmp.obs['y0'], tmp.obs['x1'], tmp.obs['y1'], segmentation[annot])
                        tmp.obs[f'frac_{annot}'] = tmp.obs.apply(lambda x: get_frac_background(x['x0'], x['y0'], x['x1'], x['y1'], segmentation[annot]), axis = 1)

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

    print("Resegmenting adata from overlapping tiles")
    adata_resegmented = resegment_h5ad(adata)
    print('...done')
    # Replace nan with 0
    for col in adata_resegmented.obs.columns:
        adata_resegmented.obs[col] = adata_resegmented.obs[col].replace(np.nan, 0)

    adata_resegmented = adata_resegmented[(adata_resegmented.obs['frac_tissue'] > .65) & (adata_resegmented.obs['frac_fold'] < .35)]

    adata_resegmented.write(f'/common/lamt2/HPV/data/foundation_models/{model}/20x_{tile_len}px_{overlap}px_overlap/h5ad_files/{slide}_combined.h5ad')

    print('---done---')