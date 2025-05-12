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

# Function for simpleitk alignment iterations
def command_iteration(method):
    """ Callback invoked when the optimization has an iteration """
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    if method.GetOptimizerIteration() % 10 == 0:
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():7.5f} "
            + f": {method.GetOptimizerPosition()}"
        )

# Function to do image alignment and save transformation
def alignimgs(fixed, moving, savepth_md, snm='', verbose=True):
 
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0,
                                                minStep=1e-4,
                                                numberOfIterations=700,
                                                gradientMagnitudeTolerance=1e-8 )

    R.SetOptimizerScalesFromIndexShift()  

    tx = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity2DTransform())
    R.SetInitialTransform(tx) 
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    outTx = R.Execute(fixed, moving)

    # if savename is given, save transformation in savepth_md directory
    if len(snm) != 0:
        savenm = f'{savepth_md}/tfm_{snm}.hdf'
        print (savenm)
        sitk.WriteTransform(outTx, savenm)
        print('saved: ', savenm)

    if verbose:
        print("-------")
        print(outTx)
        print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
        print(" Iteration: {0}".format(R.GetOptimizerIteration()))
        print(" Metric value: {0}".format(R.GetMetricValue()))

    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)
    
    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2)

    nda = sitk.GetArrayFromImage(cimg)
    
    return nda


def doit_percore_afterprealign(cid, pth_img, savepth_md, savepth_img, verbose=True):
    t0 = time.time()
    if verbose:
        print(f'reading files for sample {cid} ...', end='')

    # read images
    fixed = sitk.ReadImage(f'{pth_img}/{cid.split('/')[1]}_dapi.png', sitk.sitkFloat32)
    moving = sitk.ReadImage(f'{pth_img}/{cid.split('/')[1]}_he.png', sitk.sitkFloat32)

    if verbose:
        print(f' done: {(time.time() - t0)/60:.2f} min')
    t0 = time.time()
    # get array from sitk image
    arrf = sitk.GetArrayViewFromImage(fixed)
    arrm = sitk.GetArrayViewFromImage(moving)
    
    # Why do we go from image -> array -> image instead of just using the image?
    # Ask Yoona about this later
    moving= sitk.GetImageFromArray(arrm)
    fixed = sitk.GetImageFromArray(arrf)
    
    if verbose:
        print('aligning images ...')
    # Do the alignment
    nda = alignimgs(fixed, moving, savepth_md, cid.split('/')[1], verbose=False)
    if verbose:
        print   (f'... done: {(time.time() - t0)/60:.2f} min')
        
    t0 = time.time()
    if verbose:
        print('saving and plotting ...', end = "")
    # Save the output image as numpy array
    savenm_npy = f'{savepth_img}/{cid.split('/')[-1]}.npy'
    np.save(savenm_npy, nda)

    # Save the output images overlaid in a png
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(35,35))
    ax.imshow(nda[:,:,1], cmap='Blues_r')
    ax.imshow(nda[:,:,0], cmap='Reds_r', alpha=0.5)
    ax.set_axis_off()
    plt.title(cid)
    plt.savefig(f'{savepth_img}/images/{cid.split('/')[1]}_aligned.png')
    plt.show()
    plt.close()
    if verbose:
        print(f'  done: {(time.time() - t0) / 60:.2f} min')
    del fixed, moving, arrf, arrm
    gc.collect()
    
    return nda

s0 = int(sys.argv[1])
sf = int(sys.argv[2]) 
c = int(sys.argv[3]) if len(sys.argv) > 3 else None

# crop index file
df_idx_core = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col=0)

if sf == 0:
    runlist = [s0]
elif sf == -1:
    runlist = range(s0,len(df_idx_core['slide'].unique()))
else:
    runlist = range(s0,sf)

slides = df_idx_core['slide'].unique()[runlist]
df_idx_sub = df_idx_core[df_idx_core['slide'].isin(slides)]
print(f'running for slides {slides} :\n')

pth_img = '/common/lamt2/HPV/data/xenium/alignment_v2/prealigned_cores'
savepth_md = '/common/lamt2/HPV/data/xenium/alignment_v2/transformations'
savepth_img = '/common/lamt2/HPV/data/xenium/alignment_v2/aligned_cores'

for cid in df_idx_sub.index:
    
    # If only running one specific core: 
    if c is not None:
        core = int(cid.split('___')[-1])
        if core != c:
            continue

    nda = doit_percore_afterprealign(cid, pth_img, savepth_md, savepth_img, verbose=True)
    del nda
    gc.collect()
print('--- done ---')