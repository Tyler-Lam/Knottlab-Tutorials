
import PIL
from PIL import Image
from skimage.transform import rescale


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sys
import time
import os
import warnings


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

# Parse input arguments
s0 = int(sys.argv[1])
sf = int(sys.argv[2]) 
    
# crop index file
df_idx = pd.read_csv('/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv', index_col = 0)
df_idx[['x0', 'x1', 'y0', 'y1']] = df_idx[['x0', 'x1', 'y0', 'y1']] * 2**4
slides = df_idx['slide'].unique()

# Get list of slides to crop based on input arguments
if sf == 0:
    runlist = [s0]
elif sf == -1:
    runlist = range(s0,len(slides))
else:
    runlist = range(s0,sf)
    
print('running for slides ',slides[runlist])

for sid in runlist:
    
    # Get slide name from slide index
    slide = slides[sid]
    # Load aligned slide (saved as a numpy array)
    nda = np.load('/common/lamt2/HPV/data/xenium/alignment_v2/aligned_slides/{}.npy'.format(slide.split('/')[1]))

    # Loop over each core (index of dataframe is {slide}___{core})
    for batch in df_idx[df_idx['slide'] == slide].index:
        
        # Get the cropping coordinates
        imgidx = df_idx.loc[batch]
        
        # Crop the image using the cropping coordinates
        im_mip = Image.fromarray(nda[imgidx.y0:imgidx.y1,imgidx.x0:imgidx.x1,0])
        im_he = Image.fromarray(nda[imgidx.y0:imgidx.y1,imgidx.x0:imgidx.x1,1])
        
        # Save cropped images
        im_mip.save(f'/common/lamt2/HPV/data/xenium/alignment_v2/prealigned_cores/{batch.split('/')[-1]}_dapi.png')
        im_he.save(f'/common/lamt2/HPV/data/xenium/alignment_v2/prealigned_cores/{batch.split('/')[-1]}_he.png')

        # Plot the cropped dapi and he images
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(35,35))
        ax.imshow(nda[imgidx.y0:imgidx.y1, imgidx.x0:imgidx.x1,1], cmap='Blues_r')
        ax.imshow(nda[imgidx.y0:imgidx.y1, imgidx.x0:imgidx.x1,0], cmap='Reds_r', alpha=0.5)
        ax.set_axis_off()
        plt.title(batch)
        plt.savefig(f'/common/lamt2/HPV/data/xenium/alignment_v2/prealigned_cores/{batch.split('/')[-1]}.png')
        plt.show()
        plt.close()
        print(f'{batch} -- done')

print('done!')