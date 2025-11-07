# Python script to apply alignment transformations to annotations and save aligned images

import SimpleITK as sitk
from skimage import filters

import pyvips
# from skimage.transform import resize, rescale
import PIL
from PIL import Image
import imutils
from scipy import ndimage
from scipy.ndimage import shift
from skimage.transform import rescale

import cv2

from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc

import gc
import sys
import time
import os
import pathlib
from pathlib import Path
import warnings
from scipy.signal import fftconvolve

import tifffile

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.colors
mpl.rcParams['pdf.fonttype'] = 42

from scipy.signal import find_peaks
sc._settings.settings._vector_friendly=True


# Read in the H&E slide croppings
df_idx_slides = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_slides.csv', index_col=0)
df_idx_cores = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col=0)

LEVEL=0
pix_size =0.2125 # for level0
level_factor = 5 # for level0 (slide alignment done at level 5)
level_cropping = 4 # tiff level used for cropping alignment

# Dictionary to get the h&e slide names from the xenium output name
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

s0 = int(sys.argv[1]) # First slide
sf = int(sys.argv[2]) # last slide (non-inclusive)
cidx = int(sys.argv[3]) if len(sys.argv) > 3 else None # Core to align (align all if None)

if sf == 0:
    runlist = [s0]
elif sf == -1:
    runlist = range(s0,len(df_idx_slides))
else:
    runlist = range(s0,sf)

slides = df_idx_cores['slide'].unique()[runlist]

print('running for slides ',slides )

# These files are very large, the del and gc.collect() makes sure we're not wasting memory on objects that we don't need anymore
for slide in slides:
    
    print(f'\n\n ------------ {slide} ------------\n\n')
    
    t0 = time.time()
    print('reading in tiff files ...', end = '')
    # Read in the fixed (DAPI) image and annotations
    f_fix = f'/common/knottsilab/xenium/hpv/{slide}/morphology_focus/morphology_focus_0000.ome.tif'
    image_dapi_lv0 = tifffile.imread(f_fix, is_ome = False, level = 0)

    # Read in the fold masks
    f_tiff = f'/common/knottsilab/xenium/hpv/annotation_masks/{anno_dict[slide]}.tif'
    image_he_lv0 = tifffile.imread(f_tiff, is_ome = False, level = 0)

    print (f'   done: {(time.time() - t0)/60:.2f} min')
    
    # Get cropping coords for the DAPI image
    crop_idx = df_idx_slides.loc[slide] * (2 ** level_factor) # scale factor since cropping coordinates are for level 5 and alignment is level 0
    crop_idx = crop_idx.astype('int')

    t0 = time.time()
    print ("preprocessing ...", end = '')
    # Rotate, crop, and scale the dapi
    arr_dapi = image_dapi_lv0.astype('float32')   
    
    arr_he = np.rot90(image_he_lv0)

    arr_he = arr_he[arr_he.shape[0] - crop_idx.x1:arr_he.shape[0] - crop_idx.x0, crop_idx.y0:crop_idx.y1]

    # Scale by 1.2 but scale transformations by extra factor of 2 (recall we scaled by 0.6 to get the slide alignment)
    arr_he = rescale(arr_he.astype('float32'), 1.2, anti_aliasing=False, order=0) 
    
    del image_dapi_lv0, image_he_lv0
    gc.collect()
        
    # Invert the dapi image (idk why need to ask Yoona again)
    arr_dapi = (arr_dapi - np.min(arr_dapi)) / (np.max(arr_dapi)- np.min(arr_dapi)) * 255
    arr_dapi = 255 - arr_dapi
    
    fixed_lv0 = sitk.GetImageFromArray(arr_dapi)
    t1 = time.time()

    print(f'  done: {(t1-t0)/60:.2f} min')
    
    # Read in the alignment transformations and adjust for level 0
    modelnm = f'/common/lamt2/HPV/data/xenium/alignment_v2/aligned_slides/tfm_{slide.split('/')[1]}.hdf'
    outTx_ogto0 = sitk.ReadTransform(modelnm)

    outTx_ogto0.SetTranslation(tuple(x * (2 ** (level_factor + 1)) for x in outTx_ogto0.GetTranslation())) 
    outTx_ogto0.SetCenter(tuple(x * (2 ** (level_factor + 1)) for x in outTx_ogto0.GetCenter())) 

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_lv0)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx_ogto0)

    t2 = time.time()
    print(f'prepared sitk transformation: {(t2 - t1)/60:.2f} min')

    # Now loop over each value of possible annotations
    # Annot 1 = background
    
    #for n in np.unique(arr_he):
    for n in [1]:
        print(f'Aligning annotation value {n}\n')
        # Skip 0 annotation value (border padding)
        if (n == 0):
            continue

        # Filter all values not = annotation mask
        arr_he_annot = arr_he.copy()
        arr_he_annot[arr_he_annot != n] = 0

        # Align annotation mask
        moving_lv0 = sitk.GetImageFromArray(arr_he_annot.astype('float32'))
        out = resampler.Execute(moving_lv0)
        del moving_lv0
        gc.collect()
        
        simg = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        
        nda = sitk.GetArrayFromImage(simg)
        print(f'aligned annotation slide: {(time.time() - t2)/60:.2f} min')
        del out, simg, arr_he_annot
        gc.collect()
        
        # Now that we aligned the annotation slide, crop the core and align those individually
        for batch in df_idx_cores[df_idx_cores['slide'] == slide].index:
            
            core = int(batch.split('___')[-1])
            if cidx is not None:
                if core != cidx:
                    continue

            print(f'Cropping core {batch.split('___')[-1]}')
            imgidx = df_idx_cores.loc[batch] * (2 ** level_cropping)
            
            # Crop the core from the annotation mask
            arr_core_mask = nda[imgidx.y0:imgidx.y1, imgidx.x0:imgidx.x1]
            arr_core_mask[arr_core_mask != 0] = 1
            
            # Only do core-by-core alignment if there's something interesting in the mask
            if not np.array_equal(np.unique(arr_core_mask), [0]):
                
                t3 = time.time()
                # Load and prepare the core alignment
                modelnm_core = f'/common/lamt2/HPV/data/xenium/alignment_v2/transformations/tfm_{batch.split('/')[1]}.hdf'
                outTx_core = sitk.ReadTransform(modelnm_core)
                fixed_core = sitk.ReadImage(f'/common/lamt2/HPV/data/xenium/alignment_v2/prealigned_cores/{batch.split('/')[-1]}_dapi.png', sitk.sitkFloat32)
                
                resampler_core = sitk.ResampleImageFilter()
                resampler_core.SetReferenceImage(fixed_core)
                resampler_core.SetInterpolator(sitk.sitkLinear)
                resampler_core.SetDefaultPixelValue(0)
                resampler_core.SetTransform(outTx_core)
                
                im_msk = sitk.GetImageFromArray(arr_core_mask)
                
                # Do the alignment and save output array and image
                out = resampler_core.Execute(im_msk)
                
                simg = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
                
                t4 = time.time()
                img_arr = sitk.GetArrayFromImage(simg)
                print (f'aligned core: {(time.time() - t3)/60:.2f} min')
                del fixed_core, im_msk, out, simg
                gc.collect()

                np.save(f'/common/lamt2/HPV/data/xenium/alignment_v2/annotations/{batch.split('/')[1]}_annot{n}.npy', img_arr)
                print(f'   saved aligned annotation: {(time.time() - t4)/60:.2f} min')

                del img_arr
                gc.collect()

            else:
                print(f'   No annotations found')
            
            del arr_core_mask
            gc.collect()
                
        del nda
        gc.collect()
    
    del arr_dapi, fixed_lv0, arr_he
    gc.collect()
    
print('done!')