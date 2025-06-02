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

mpp_lv0 = 0.262774

s0 = int(sys.argv[1])
sf = int(sys.argv[2]) 
cidx = int(sys.argv[3]) if len(sys.argv) > 3 else None

# crop index file
df_idx_slides = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_slides.csv', index_col=0)
df_idx_cores = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col=0)

if sf == 0:
    runlist = [s0]
elif sf == -1:
    runlist = range(s0,len(df_idx_cores['slide'].unique()))
else:
    runlist = range(s0,sf)
    
LEVEL=1 # Level of h&e images to align/crop
pix_size =0.2125 * (2**LEVEL) # for level0
level_factor = 5 - LEVEL # for level0 (slide alignment done at level 5)
level_cropping = 4 - LEVEL # tiff level used for cropping alignment

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

slides = df_idx_cores['slide'].unique()[runlist]


# These files are very large, the del and gc.collect() makes sure we're not wasting memory on objects that we don't need anymore
for slide in slides:
    
    wsis = []
    mpps = []
    mpp_scales = []
    
    print(f'\n\n ------------ {slide} ------------\n\n')
    
    t0 = time.time()
    print('reading in tiff files', end = '')
    # Read in the fixed (DAPI) image and annotations
    f_fix = f'/common/knottsilab/xenium/hpv/{slide}/morphology_focus/morphology_focus_0000.ome.tif'
    arr_dapi_lv0 = tifffile.imread(f_fix, is_ome = False, level = LEVEL)

    # Read in the he 
    f_tiff = f'/common/knottsilab/xenium/hpv/svs/{anno_dict[slide]}.tiff'
    arr_he_lv0 = tifffile.imread(f_tiff, is_ome = False, level = LEVEL)

    print (f' ... done: {(time.time() - t0)/60:.2f} min')
    
    # Get cropping coords for the DAPI image
    crop_idx = df_idx_slides.loc[slide] * (2 ** level_factor) # scale factor since cropping coordinates are for level 5 and alignment is level 1
    crop_idx = crop_idx.astype('int')

    t0 = time.time()
    print ("preprocessing", end = '')
    arr_dapi = arr_dapi_lv0.astype('float32')   
    
    # Rotate and crop the h&e
    arr_he = np.rot90(arr_he_lv0)
    arr_he = arr_he[arr_he.shape[0] - crop_idx.x1:arr_he.shape[0] - crop_idx.x0, crop_idx.y0:crop_idx.y1]
    # Scale by 1.2 but scale transformations by extra factor of 2 (since we scaled by 0.6 to get the slide alignment for levels 5-6)
    arr_he = rescale(arr_he.astype('float32'), 1.2, anti_aliasing=False, order=0, channel_axis = 2) 

    del arr_dapi_lv0, arr_he_lv0
    gc.collect()

    fixed_lv0 = sitk.GetImageFromArray(arr_dapi)
    t1 = time.time()

    print(f' ... done: {(t1-t0)/60:.2f} min')

    print(f'Performing slide alignment', end = '')
    t2 = time.time()
    
    # Read in the slide alignment transformations and adjust for level 1
    modelnm = f'/common/lamt2/HPV/data/xenium/alignment_v2/aligned_slides/tfm_{slide.split('/')[1]}.hdf'
    outTx_ogto0 = sitk.ReadTransform(modelnm)

    outTx_ogto0.SetTranslation(tuple(x * (2 ** (level_factor + 1)) for x in outTx_ogto0.GetTranslation())) 
    outTx_ogto0.SetCenter(tuple(x * (2 ** (level_factor + 1)) for x in outTx_ogto0.GetCenter())) 

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_lv0)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx_ogto0)

    # Explaination of mpp scaling: (In theory it should be pix_size to match the xenium level, but I want to be precise)
    # Base level 0 h&e mpp = 0.262774
    # Every level is 2x downsampled (increases mpp)
    # rescale of 1.2 --> 1.2 times more pixels --> decreases mpp
    # sitk scale < 0 --> More pixels --> increases mpp
    # sitk scale > 0 --> Fewer pixels --> decreases mpp
    mpp_slide = (mpp_lv0 * (2 ** LEVEL)) * outTx_ogto0.GetScale() / 1.2

    aligned_he = []
    for i in range(3):
        tmp = sitk.GetImageFromArray(arr_he[:,:,i].astype('float32'))
        aligned_he.append(sitk.GetArrayFromImage(resampler.Execute(tmp)))
        del tmp
        gc.collect()

    # Transpose to make the array format match the RGB image formats
    aligned_he = np.transpose(np.array(aligned_he), (1, 2, 0))
    print(f' ... done: {(time.time() - t2)/60:.2f} min')

    del arr_he, fixed_lv0
    gc.collect()
    
    # Now that we aligned the annotation slide, crop the core and align those individually
    for batch in df_idx_cores[df_idx_cores['slide'] == slide].index:
        
        core = int(batch.split('___')[-1])
        if cidx is not None:
            if core != cidx:
                continue

        print(f'Aligning core {batch.split('___')[-1]}', end = '')
        t3 = time.time()
        imgidx = df_idx_cores.loc[batch] * (2 ** level_cropping)
        
        # Crop the core from the annotation mask
        arr_core = aligned_he[imgidx.y0:imgidx.y1, imgidx.x0:imgidx.x1]
        
        modelnm_core = f'/common/lamt2/HPV/data/xenium/alignment_v2/transformations/tfm_{batch.split('/')[1]}.hdf'
        outTx_core = sitk.ReadTransform(modelnm_core)
        outTx_core.SetTranslation(tuple(x * (2 ** (0 - LEVEL)) for x in outTx_core.GetTranslation())) 
        outTx_core.SetCenter(tuple(x * (2 ** (0 - LEVEL)) for x in outTx_core.GetCenter())) 

        fixed_core = sitk.GetImageFromArray(arr_dapi[imgidx.y0:imgidx.y1, imgidx.x0:imgidx.x1])

        resampler_core = sitk.ResampleImageFilter()
        resampler_core.SetReferenceImage(fixed_core)
        resampler_core.SetInterpolator(sitk.sitkLinear)
        resampler_core.SetDefaultPixelValue(0)
        resampler_core.SetTransform(outTx_core)

        # Do the alignment and save output array and image
        out_arr = []
        for i in range(3):
            tmp = sitk.GetImageFromArray(arr_core[:,:,i].astype('float32'))
            out_arr.append(sitk.GetArrayFromImage(resampler_core.Execute(tmp)))
            del tmp
            gc.collect()
            
        # Make array match rgb format arr[x,y] = [r,g,b]
        out_arr = np.transpose(np.array(out_arr), (1, 2, 0))
        
        # Scale the image to the svs level 1 mpp (In theory these values should be 2*mpp_lvl0 and 0.8086797, respectively)
        mpp_core = (mpp_slide ) * outTx_core.GetScale()
        mpp_scale = mpp_core / (2.0 * mpp_lv0)    
                
        mpps.append(mpp_core / mpp_scale)
        wsis.append(f'{anno_dict[slide]}___{core}.png')
        
        # Also save the scale so we can re-scale to match the dapi
        mpp_scales.append(mpp_scale)
        
        out_arr = rescale(out_arr.astype('float32'), mpp_scale, anti_aliasing=False, channel_axis = 2)
        
        print(f' ... done: {(time.time() - t3)/60:.2f} min')
        t4 = time.time()
        print(f'Saving image to /common/knottsilab/xenium/hpv/aligned_cores/{anno_dict[slide]}___{core}.png', end = '')
        out_png = Image.fromarray(out_arr.astype(np.uint8), mode = 'RGB')
        out_png.save(f'/common/knottsilab/xenium/hpv/aligned_cores/{anno_dict[slide]}/{anno_dict[slide]}___{core}.png')
        print(f' ... done: {(time.time() - t4)/60:.2f} min')
        

        del fixed_core, out_arr, out_png, arr_core
        gc.collect()

    del aligned_he
    gc.collect()
    
    t4 = time.time()
    df = pd.DataFrame({'wsi': wsis, 'mpp': mpps, 'mpp_scale': mpp_scales})
    df = df.set_index('wsi')
    if os.path.exists(f'/common/knottsilab/xenium/hpv/aligned_cores/{anno_dict[slide]}/resolutions.csv'):
        resolutions = pd.read_csv(f'/common/knottsilab/xenium/hpv/aligned_cores/{anno_dict[slide]}/resolutions.csv', index_col = 0)
        df = pd.concat([df, resolutions[~resolutions.index.isin(df.index)]], join = 'outer', axis = 0)
    df.to_csv(f'/common/knottsilab/xenium/hpv/aligned_cores/{anno_dict[slide]}/resolutions.csv')

print('done!')