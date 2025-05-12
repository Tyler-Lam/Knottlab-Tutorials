from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scanpy as sc

from PIL import Image

import gc
import time
import sys

import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.colors
mpl.rcParams['pdf.fonttype'] = 42

sc._settings.settings._vector_friendly=True

slide_idx = [
    '20250305__223640__X206_03052025_HPVTMA_01_02/output-XETG00206__0060366__Region_1__20250305__223715',
    '20250305__223640__X206_03052025_HPVTMA_01_02/output-XETG00206__0060364__Region_1__20250305__223715',
    '20250213__202616__X206_02132025_ANOGENTMA_1_2/output-XETG00206__0060075__Region_1__20250213__202651',
    '20250213__202616__X206_02132025_ANOGENTMA_1_2/output-XETG00206__0060077__Region_1__20250213__202651',
    '20250224__233848__X206_2242025_ANOGENTMA_03_04/output-XETG00206__0060367__Region_1__20250224__233922',
    '20250224__233848__X206_2242025_ANOGENTMA_03_04/output-XETG00206__0060354__Region_1__20250224__233922',
    '20250304__005745__X403_03032025_ANOGENTMA_05_06/output-XETG00403__0060395__Region_1__20250304__005817',
    '20250304__005745__X403_03032025_ANOGENTMA_05_06/output-XETG00403__0059911__Region_1__20250304__005817',
    '20250312__003942__X206_03112025_HPVTMA_03_04/output-XETG00206__0060488__Region_1__20250312__004017',
    '20250312__003942__X206_03112025_HPVTMA_03_04/output-XETG00206__0060493__Region_1__20250312__004017'
]

s0 = int(sys.argv[1]) # First slide
sf = int(sys.argv[2]) # last slide (non-inclusive)
cid = int(sys.argv[3]) if len(sys.argv) > 3 else None

if sf == 0:
    runlist = [s0]
elif sf == -1:
    runlist = range(s0,len(slide_idx))
else:
    runlist = range(s0,sf)

# Read in cropping coordinates for the cores
df_idx = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col=0)
t0 = time.time()
# Read in the anndata
adata = sc.read_h5ad('/common/lamt2/HPV/data/xenium/adata/adata_coreograph.h5ad')
adata.obs['batch'] = adata.obs['slide'].astype(str) + '___' + adata.obs['core_idx'].astype(str)

print (f'Read anndata in {(time.time() - t0)/60:.2f} min')

# Get all files that we have a mask for
fmasks = glob('/common/lamt2/HPV/data/xenium/alignment_v2/annotations/*.npy')

from collections import defaultdict
df_masks = defaultdict(list)

for sid in runlist:
    # Loop over aligned masks
    for batch in adata[adata.obs['slide'] == slide_idx[sid]].obs['batch'].unique():
        slide = batch.split('/')[-1].split('___')[0]
        core = int(batch.split('___')[1])
        if cid is not None:
            if cid != core:
                continue
        core_masks = list(filter(lambda x: f'{batch.split('/')[-1]}_' in x, fmasks))
        if len(core_masks) == 0:
            continue
        print (f'\n\n-----{batch}-----\n\n')

        he_arr = np.load(f'/common/lamt2/HPV/data/xenium/alignment_v2/aligned_cores/{slide}___{core}.npy')
        obs = adata.obs[adata.obs['batch'] == batch]

        for f in core_masks:
            to = time.time()
            annot = f.split('_')[-1].split('.')[0]

            print(f'Transferring {annot}')
            print('Loading mask ...', end = '')
            mask = np.load(f)
            # Temporary hack for old fold mask formatting (used to save mask[:,:,0] = dapi, mask[:,:,1] = mask)
            if len(mask.shape) > 2:
                mask = mask[:,:,1]
            print (f' done: {(time.time() - t0)/60:.2f} min')
            t0 = time.time()
            print("Applying and saving annotations ...", end = '')
            xs = obs['x_centroid'].values
            ys = obs['y_centroid'].values
            
            core_idx = df_idx[(df_idx['slide'].str.contains(slide)) & (df_idx['core'] == core)].iloc[0]
            core_labels = []

            sf = 0.2125
            LEVEL = 4 # Level used for core cropping coordinates
            for x,y in zip(xs, ys):
                xscale = int(x/sf - core_idx.x0 * (2**LEVEL))
                yscale = int(y/sf - core_idx.y0 * (2**LEVEL))
                if (xscale >= mask.shape[1] or yscale >= mask.shape[0] or xscale < 0 or yscale < 0):
                    core_labels.append(0)
                else:
                    core_labels.append(mask[yscale,xscale])
            
            obs.insert(loc = len(obs.columns), column = annot, value = core_labels)
            obs[[annot]].to_pickle(f'/common/lamt2/HPV/data/xenium/alignment_v2/annotations/{slide}___{core}_{annot}.pkl')
            df_masks[annot].append(obs[[annot]])
            print(f'  done: {(time.time() - t0)/60:.2f} min')
            t0 = time.time()
            
            print ("Plotting cells. ...", end = '')
            # Now to plot everything
            f, ax = plt.subplots(1, 1, figsize = (35, 35))
            
            # Plot h&e
            ax.imshow(he_arr[:,:,1], cmap = "Blues_r")
            
            # Plot dapi
            ax.imshow(he_arr[:,:,0], cmap = 'Reds_r', alpha = 0.5)
            
            # Plot mask
            ax.imshow(mask, cmap = "Grays", alpha = 0.15)
            
            # Plot cells (masked and unmasked separated)
            xs_unmasked = obs[obs[annot] == 0]['x_centroid']/sf - core_idx.x0 * (2**LEVEL)
            ys_unmasked = obs[obs[annot] == 0]['y_centroid']/sf - core_idx.y0 * (2**LEVEL)
            xs_masked = obs[obs[annot] > 0]['x_centroid']/sf - core_idx.x0 * (2**LEVEL)
            ys_masked = obs[obs[annot] > 0]['y_centroid']/sf - core_idx.y0 * (2**LEVEL)

            ax.scatter(
                xs_unmasked,
                ys_unmasked,
                s = 30,
                c = 'Black',
                label = 'Unmasked'
            )
            ax.scatter(
                xs_masked,
                ys_masked,
                s = 30,
                c = 'C1',
                label = 'Masked'
            )
            ax.set_title(f'{slide}___{core} - {annot}', fontsize = 50)
            ax.legend()
            plt.savefig(f'/common/lamt2/HPV/data/xenium/alignment_v2/annotations/images/{slide}___{core}_{annot}.png')
            plt.close()
            print(f'  done: {(time.time() - t0)/60:.2f} min')
            
            del mask
            gc.collect()
        
        del he_arr, obs
        gc.collect()

print('---done---')