# Celltyping analysis pipeline

Last updated 5/13/25 - Tyler Lam

These scripts were used for celltyping in the HPV project. They assume that you already have applied QC to the data and calculated the latent representation using scVI or equivalent.

## Scripts

1) `run_celltyping.py` - Python script to perform leiden clustering and get the differential expression for genes in each cluster
   * Run syntax: `python run_celltyping.py [-a adata_name] [-s scvi_name]`
      * Needs the anndata saved as `adata_{adata_name}.h5ad`
      * Needs the latent representation saved as `rep_{scvi_name}.npy`
   * First makes the neighborhood graph with default settings (n_nearest = 15)
   * Then calculates the UMAP with parameters provided by Aagam
   * Then does leiden clustering and diffex calculation for 3 resolutions: 0.5, 0.7, 1.0
      * We do 3 resolutions because its difficult to know a priori what the "best" resolution will be
      * Usually you want it more fine (higher) if you need to separate out mixed clusters
2) `plot_celltyping.ipynb` - Notebook to make celltyping plots to send to Joe
   * Requires that you have already done the neighborhood graph, umap, etc. from (1)
   * Outputs the following for each resolution:
      * UMAP of the leiden clusters. Occasionally Joe might ask for the umap for various genes
      * csv with the average number of cells, average number of genes expressed, and average total counts for each cluster
      * csv of the top 10 differentially expressed genes by wilcoxon score
      * csv of the top 20 highly expressed genes by total counts  <-- This one is the most useful for Joe
      * Dotplot of the top 5 diffex genes expressed in >20% of cells
   * Joe occasionally asks for dotplots of specific genes, usually when classifying subclusters of already typed cells. A few example blocks are included for reference
3) `plot_clusters.ipynb` - Notebook to plot specific clusters over the aligned h&e images
   * Uses the alignment transformations to plot specific clusters from celltyping. We used this for epithelial but it can work for any anndata that has been leiden clustered.
   * For each cluster, it plots the highest n cores by total cell count and highest proportion of cells from that cluster for the invasive and in-situ cores


## General workflow

The naming conventions I used helped me keep track of things but your mileage may vary. There's probably a better way to do this. Follow at your own risk.

1) Initial celltyping
   * We first calculate the latent representation for the entire dataset and make all the plots using (1) and (2)
   * Joe gives us the first round of primary celltypes as either Immune, Stromal, Epithelial, or NOS (nonspecific, cannot be determined)
   * This is the first round of primary celltyping (using the entire dataset), so I usually save the anndata as `primary_v0`
2) Iterative celltyping
   * We then separate our data into each primary celltype and calculate the latent representation so we can remake celltyping plots for each one. Per Aagam we used the same scVI architecture that we used for the full data set
      * There will usually be a few iterations of reclassifying clusters
      * Everytime you add/remove clusters from a celltype, you have to redo the latent rep/clustering
      * I try to name the anndatas to keep track of the iterations. For example, the first version of immune cells (without reshuffling/reclustering) is usually `primary_v0_Immune_v0`. If we shuffle it by adding/removing another cluster, it would be saved as `primary_v0_Immune_v1`.
   * Once the primary clustering is stable (ie no more shuffling primary celltypes), Joe will classify the primary celltypes into what I've been calling secondary "supertypes" (not yet fully called secondary celltypes/mixes of secondary celltypes)
      * Here are the types that we had for the HPV project:
         * Immune: TNK, Mast cell, Myeloid, B/Plasma
         * Stromal: Fibroblast, Endothelial_Pericyte, Neural, Muscle
         * Epithelial: Unable to be typed based on gene expression alone, needed spatial plots over the h&e (see `plot_clusters.ipynb`)
      * Each secondary type is separated and the clustering pipeline is redone
      * These will also likely need to be shuffled around and reclustered
      * Similar naming conventions to previous. Say we got our first classification of Myeloid cells from the `primary_v0_Immune_v1` clustering. Then the myeloid cells would be saved as `primary_v0_Immune_v1_Myeloid_v0`

Basically, the naming version is based on which group of cells we recluster. If we recluster all cells from the full dataset, we would iterate to `primary_v{n+1}`. If we recluster all Stromal cells, we iterate to `primary_vX_Stromal_v{n+1}`. And if we recluster B_Plasma cells, we would iterate to `primary_vX_Immune_vY_B_Plasma_v{n+1}`.

If this is confusing thats cause this is probably a bad naming convention. However its very useful to be able to keep track of which cells moved in which iterations so you can keep track of where shuffled cells get clustered.