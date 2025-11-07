# Foundation Model Analysis Pipeline

Last updated 6/2/25 - Tyler Lam

Scripts used to extract features from the foundation model outputs. These models take h&e images as input, divide them into small patches, and output an embedding for each patch. Current best practice is to run at 20x magnification with 50% overlap (112 pixels for the Virchow2 model) and extract the patch/cls embeddings. See [here](https://cedarssinai-my.sharepoint.com/:p:/g/personal/tyler_lam_cshs_org/Ec8wgUzircNOgvC_ZdjWY8UBc1FhziP2gjD8Kyj83Xsucw?e=EBC5xX) for an overview of patch/tile embeddings

### Scripts

Note: Most sources use "magnification" imprecisely without specifying a base resolution. For our h&e images, the highest resolution (level 0) is 0.262774 microns per pixel (mpp) corresponding to 40x magnification.

1. `run_foundation_model_alignment.py` - Python script to get the h&e images that will be used as input to the foundation models
    * This script assumes that you have already calculated the level 0 alignment transformations from sitk and annotation masks (in .npy format)
    * It is very similar to the alignment script with a few key differences:
        * Original alignment only uses the "blue" channel from the rgb h&e images. These scripts need the full multi-channel images to work with the foundation models
        * Foundation models largest training set is 20x magnification, so we need to adjust our alignments to work with level-1 h&e
        * This script saves the resolution (in mpp) and scale factor to get from dapi level 0 to 20x magnification
    * Outputs core images as png files
2. `modify_foundation_model_segmentation.ipynb` - Notebook to redo foundation model segmentation
    * Foundation models do segmentation to identify tissue automatically, but its usually very coarse
    * We use the detailed annotation mask to identify the fraction of tissue/fold in each patch
    * Outputs the segmentation for each core as a geojson file
3. To actually run the foundation models see `/common/knottsilab/MET_analysis/h_and_e_analysis/run_foundation_models/sub_trident_job_virchow.sh` as an example of how to submit jobs
    * With png files, you need an additional argument `--custom_list_of_wsis "/path/to/resolution.csv"`
    * Currently only the Virchow2 model is configured to return small patch embeddings instead of just full tile embeddings
4. `process_foundation_model_outputs.py` - Python script to merge foundation model feature outputs for each slide and apply to cells from xenium data
    * First aggregates all cores for a given slide into one anndata
        * If patches have 50% overlap and individual patch embeddings, we weight the contributions from each tile [as shown here](https://cedarssinai-my.sharepoint.com/:p:/g/personal/tyler_lam_cshs_org/EUFOhn9spYRGhmS7aEEsTg8BQMRYpyGOVqPkW_AfOLffvQ?e=GLD2ad)
    * Then we apply the patch embeddings to cells in the xenium anndata. Note if submitting this script as a batch job to increase --cpus-per-task, or this will take a long time to run