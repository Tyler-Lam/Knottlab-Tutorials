# Python script to apply alignment transformations to H&E slides and save aligned images

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
df_idx = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment/cropped_slides.csv', index_col=0)

LEVEL=0
pix_size =0.2125 # for level0
level_factor = 5 # for level0 (slide alignment done at level 5)

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

if sf == 0:
    runlist = [s0]
elif sf == -1:
    runlist = range(s0,len(df_idx))
else:
    runlist = range(s0,sf)

print('running for slides ',df_idx.iloc[runlist].index.values )

# These files are very large, the del and gc.collect() makes sure we're not wasting memory on objects that we don't need anymore
for sid in runlist:
    
    # Get the slide name from the index (index = numerical value, slide name = unique string to identify specific slide)
    slide = df_idx.iloc[sid].name
    
    print(f'\n\n ------------ {slide} ------------\n\n')
    
    t0 = time.time()
    # Read in the fixed (DAPI) image and moving (H&E) image
    f_fix = f'/common/knottsilab/xenium/hpv/{slide}/morphology_focus/morphology_focus_0000.ome.tif'
    image_dapi_lv0 = tifffile.imread(f_fix, is_ome = False, level = 0)

    f_tiff = f'/common/knottsilab/xenium/hpv/svs/{anno_dict[slide]}.tiff'
    image_he_lv0 = tifffile.imread(f_tiff, is_ome = False, level = 0)

    print (f' read in tiff files in {(time.time() - t0)/60:.2f} min')
    
    # Get cropping coords for the DAPI image
    crop_idx = df_idx.iloc[sid] * (2 ** level_factor) # scale factor since cropping coordinates are for level 5 and alignment is level 0
    crop_idx = crop_idx.astype('int')

    t0 = time.time()
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

    print(f'preprocessing is done: {(t1-t0)/60:.2f} min')
    
    # Read in the alignment transformations and adjust for level 0
    modelnm = f'/common/lamt2/HPV/data/xenium/alignment/aligned_slides/tfm_{slide.split('/')[1]}.hdf'
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
    
    # every [x,y] coord in image array has value = [r,g,b,alpha=255]
    # align each value separately and then combine (also need to ask Yoona why)
    out = []
    for i in range(3):
        tmp = sitk.GetImageFromArray(arr_he[:,:,i].astype('float32'))
        out.append(resampler.Execute(tmp))
        del tmp
        gc.collect()
        
    # Combine the images into one array and save
    # (need to get details from Yoona about why)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed_lv0), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out[0]), sitk.sitkUInt8)
    simg3 = sitk.Cast(sitk.RescaleIntensity(out[1]), sitk.sitkUInt8)
    simg4 = sitk.Cast(sitk.RescaleIntensity(out[2]), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg3, simg4)
    
    nda = sitk.GetArrayFromImage(cimg)

    t3 = time.time()
    print(f'alignment in high resolution per slide done: {(t3 - t2)/60:.2f} min')
    
    np.save(f'/common/lamt2/HPV/data/xenium/alignment/aligned_slides/{slide.split('/')[1]}.npy', nda)
    t4 = time.time()
    print(f'alignment saved: {(t4 - t3)/60:.2f} min')


    # To troubleshoot if we notice problems later: save the aligned full resolution image
    # Change False -> True to execute the block below
    if False:
        f, axs = plt.subplots(nrows=1, ncols=3)
        axs[0].imshow(nda[:,:,0], cmap='Blues')
        axs[0].set_title('dapi image')
        
        axs[1].imshow(nda[:,:,1], cmap='Blues')
        axs[1].set_title('H&E image')
        
        axs[2].imshow(nda[:,:,2], cmap='Blues')
        axs[2].set_title('aligned')
        plt.savefig('/common/lamt2/HPV/data/xenium/alignment/aligned_slides/aligned_{}.png'.format(slide.split('/')[1]))
        #plt.show()
        
        t5 = time.time()
        print(f'alignment in high resolution per slide image is saved: {(t5-t4)/60} min')
    
    del out, simg1, simg2, simg3, simg4, nda, cimg, fixed_lv0, arr_dapi, arr_he
    gc.collect()

print('done!')