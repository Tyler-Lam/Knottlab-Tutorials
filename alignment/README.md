# Alignment and annotation analysis pipeline

Last updated 5/12/25 - Tyler Lam

These scripts were written initially for the HPV project and are heavily adapted from Yoona's alignment code. The raw data, coreograph outputs, h&e stains, and annotation masks are located in the `/common/knottsilab/xenium/hpv` directory. The purpose is to align the dapi images to the h&e stains in order to associate annotated regions with cells in the data.

## Scripts

This is a multi-step process to get accurate alignment of the dapi and h&e using simpleITK (sitk). First we do a rough alignment of the full slides, then we crop the individual cores from the slide-level alignment and individually calculate alignment transformations for each core. Then we apply the full alignment transformation to the annotation masks. Scripts should be run in the following order. Output directories have `alignment_v2` because I had to redo alignment with Joe's annotation labels.

1) `align_he.ipynb` - Jupyter notebook used to align the dapi and h&e using the full slide at low resolution
   * First we crop the h&e slide, since there is usually a large border/padding around the edges. These cropping coords are saved to `/common/lamt2/HPV/data/xenium/alignment_v2/cropped_slides.csv`
   * Then we do rough alignment of the slides and a quick visual inspection. The transformations are saved to `/common/lamt2/HPV/data/xenium/alignment_v2/aligned_slides/tfm_{slide}.hdf`
   * dapi is read at level 6, h&e is read at level 5 and scaled by 0.6 as a first approximation
   * __Core labeling__
      * At this point, if you have run coreograph and assigned core labels to each cell, you can proceed to the next step
      * If you are using annotation labels from Joe instead, you have to apply the slide alignment transformations to the annotation labels as seen in `run_TMA_label_alignment.py` and apply the new labels to the anndata
2) `crop_corebycore.ipynb` - Jupyter notebook to crop each core using level 4 dapi image
   * Uses cell coordinates + padding as a first estimate
   * Cropping coordinates are saved to `/common/lamt2/HPV/data/xenium/alignment_v2/cropped_cores.csv`
   * If you trust the annotation masks, you can skip the manual cropping and just assign the indices using the first estimate. However, sometimes the dapi and h&e have features that don't produce actual cells in the anndata. You may have to go back to this step later and re-crop those cores.
3) `run_slide_alignment.py` - Script to do full slide alignment in high resolution
   * Takes full-slide transformations calculated in (1) (slide cropping + rescaling + sitk transformation) and applies them for the full resolution h&e images
   * Images are saved as numpy arrays in `/common/lamt2/HPV/data/xenium/alignment_v2/aligned_slides/{slide}.npy`
   * Run syntax: `python run_slide_alignment.py slide_idx_start slide_idx_end`
      * `slide_idx_start` is the index of the first slide to align
      * `slide_idx_end` is the index of the last slide to align (non-inclusive). Set to 0 to only run on `slide_idx_start` or -1 to run on all slides after `slide_idx_start`
4) `run_crop_corebycore.py` - Python script to crop cores from aligned full-resolution slide images
   * Uses cropping coordinates calculated in (2), scaled from tiff level 4 to tiff level 0
   * Saves dapi, h&e, and overlaid images to `/common/lamt2/HPV/data/xenium/alignment_v2/prealigned_cores/{slide}___{core}{_he/dapi}.npy`
   * Run syntax same as (3): `python run_crop_corebycore.py slide_idx_start slide_idx_end`
5) `run_core_alignment.py` - Python script to align the cropped core images from (4)
   * Saves the transformations for core alignment to `/common/lamt2/HPV/data/xenium/alignment_v2/transformations/tfm_{slide}___{core}.hdf`
   * Saves the aligned images as png files to `/common/lamt2/HPV/data/xenium/alignment_v2/aligned_cores/images/{slide}___{core}_aligned.png`
   * Saves the aligned images as np arrays to `/common/lamt2/HPV/data/xenium/alignment_v2/aligned_cores/{slide}___{core}.npy`
   * Run syntax: `python run_core_alignment.py slide_idx_start slide_idx_end [core_idx]`
      * slide arguments same as previous
      * [core_idx] is an optional parameter. If included, only run alignment on the specified core. Otherwise align all cores
6) `run_annotation_alignment.py` - Python script to run the full alignment on the annotation masks from `/common/knottsilab/xenium/hpv/annotation_masks/*.tif`
   * First crops the slides and does full slide alignment using coordinates and transformations from (1)
   * Then crops each core using the coordinates from (2)
   * Then aligns each core using transformations from (5)
   * Aligned annotation mask is saved as a numpy array in `/common/lamt2/HPV/data/xenium/alignment_v2/annotations/{slide}___{core}_{annotation}.npy`
   * Currently set to work on annotation masks, but can be easily modified to work on fold masks from `/common/knottsilab/xenium/hpv/fold_masks/*.tif`. Set the output suffix to be `_foldmask.npy` to work with next steps. See `run_fold_mask_alignment.py` for a quick/sloppy example.
   * Run syntax same as (5): `python run_annotation_alignment.py slide_idx_start slide_idx_end [core_idx]`
7) `run_annotation_transfer.py` - Takes the aligned annotations/fold masks and assigns labels to each cell
   * For each core, save a .pkl file of each cell index and annotation value to `/common/lamt2/HPV/data/xenium/alignment_v2/annotations/{slide}___{core}_{annotation}.pkl`. These can be merged with the original anndata
   * For each annotation value, save an image with the dapi, aligned h&e, masked cells, and unmasked cells to `/common/lamt2/HPV/data/xenium/alignment_v2/annotations/images/{slide}___{core}_{annotation}.png`
   * Run syntax same as (5): `python run_annotation_transfer.py slide_idx_start slide_idx_end [core_idx]`
8) `check_alignment.ipynb` - Jupyter notebook to check each core alignment
   * Loops over all alignment images, displaying the aligned dapi/h&e from (5) and background mask images (`_annot1` files) from (7) (and fold masks if they exist)
   * Asks for input from each image to fill dataframe with alignment level, presence of folds, and other comments/notes
      * If a core failed to align properly, see next step
      * Cores that shifted between dapi and h&e staining and need new masks should be sent to Joe to be redone
   * This code is very jank because plt.imshow() and input were being weird together
   * Dataframe is saved to `/common/lamt2/HPV/data/xenium/alignment_v2/alignment_validation.csv`
9) `do_manual_alignment.ipynb` - Jupyter notebook to align cores that were not well aligned previously
    * Using dataframe from (9), fix cores that were marked as poorly aligned by doing things manually (Can do any combination of these)
       * Gets user input to manually adjust saturation levels
       * Manually crop image to isolate parts that seem easier to align
       * Set initial offset/translation to help SimpleITK alignment
    * If image is aligned, overwrite the transformation, numpy array, and overlaid image in same directories as (5)
    * If a core is realigned this way, you have to rerun steps (6)-(7) to redo the annotation alignment/transfer (__Important__ Make sure you indicate that the annotation alignment has to be redone when editing the validation dataframe. I like to set aligned = -1 so they're easy to query)
