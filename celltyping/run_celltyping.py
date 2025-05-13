import scanpy as sc
import pandas as pd
import numpy as np
from glob import glob
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--adata', default = 'qc') # Name of anndata. Assumed to be saved as adata_{name}.h5ad
parser.add_argument('-s', '--scvi', default = 'latent_17_layers_4') # Name of scvi rep, assumed to be saved as rep_{name}.npy
args = vars(parser.parse_args())

# Input and output directories (same in my case but not always in general)
read_dir = '/common/lamt2/HPV/data/xenium/adata'
out_dir = '/common/lamt2/HPV/data/xenium/adata'

# Read in the anndata
t0 = time.time()
adata = sc.read_h5ad(f"{read_dir}/adata_{args['adata']}.h5ad")

sfx = "" if args['adata'] == 'qc' else f'_{args['adata']}'
# Read in sketched cells
sketched_cells = pd.read_pickle(f'{read_dir}/cells_sketched_per_slide{sfx}.pkl')
adata.obs['sketched'] = adata.obs.index.isin(sketched_cells)

# Read in latent representation
latent = np.load(f'{read_dir}/rep{sfx}_{args["scvi"]}.npy')
adata.obsm['X_scVI'] = latent

print(f'read adata {(time.time() - t0)/60:.2f} min')

# Make NN graph (currently using default nearest neighbors = 15)
t0 = time.time()
sc.pp.neighbors(adata, use_rep = "X_scVI")
print (f'made neighborhood graph {(time.time() - t0)/60:.2f} min')

# Make the umap, arguments from Aagam
t0 = time.time()
umap_args = {
    'min_dist': 0.1,
    'spread': 1,
    'init_pos': 'pca',
}
sc.tl.umap(adata, **umap_args)
print(f'made umap {(time.time() - t0)/60:.2f} min')

# Do the leiden clustering in 3 different resolutions
for resolution in [0.5, 0.7, 1.0]:
    res = str(resolution).replace('.', 'p')
    t0 = time.time()
    sc.tl.leiden(
        adata,
        resolution = resolution,
        random_state = 42,
        flavor = "igraph",
        n_iterations = 2,
        key_added = f'leiden_{res}',
        directed = False
    )
    print(f'leiden clustered resolution {resolution}: {(time.time() - t0)/60:.2f} min')

    # Ranking gene groups by wilcoxon
    t0 = time.time()
    sc.tl.rank_genes_groups(adata, f"leiden_{res}", method = 'wilcoxon', pts = True, key_added = f'rank_genes_groups_{res}')
    print(f'Ranked gene groups {(time.time() - t0)/60:.2f} min')

# Write the anndata
adata.write_h5ad(f"{out_dir}/adata_leiden_{args['adata']}_{args['scvi']}.h5ad")
print('---done---')