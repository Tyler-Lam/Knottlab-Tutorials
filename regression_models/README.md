# Regression Model Analysis Pipeline

Last updated 1/13/26 - Tyler Lam

Scripts used to compare features using various regression models. These models take feature dataframes and construct GLMs/regression models to extract p-values using contrast vectors.

We have two main models that we use to compare features: Beta regression (when feature are bound between 0-1), and gamma regression (when features are bound between 0-inf). The reason we use these models instead of bernoulli or poisson is due to the overdispersion from interpatient variance. Essentially, the variance between patients is much higher than the variance from either of these distributions.

We use the beta distribution to model proportions/fractions from continuous counts. Celltype proportions are given as n_cells / n_total_cells, while the covering fractions are measured in terms of n_pixels / n_total_pixels, which are discrete counts. Since these can contain exact 0/1 values, we apply the [Smithson–Verkuilen correction](https://pubmed.ncbi.nlm.nih.gov/16594767/).

Count density (n_cells / area) and 1+spatial correlations (n_exp / n_obs) can be modeled with gamma distributions (continuous) or discrete (zero-inflated) negative binomial distributions. For the gamma distribution, there is no similar method to correct exact 0 numerators, but since the numerators are discrete counts we adjust exact 0 counts to 0.1. TBD if this is correct but in general it produces reasonable results. For more precise fitting, the discrete methods can be used but take longer to run. A method of moments estimate of fit parameters is used to calculate the predicted zero counts and compared to the observed. If there are significantly more observed zeros than expected, the feature is added using a zero-inflated negative binomial.

Once all models are fit, we perform permutation testing to calculate empirical FDR values. The initial test to extract significant features uses the log likelihood ratio with a reduced model as a test statistic. This effectively measures how much better we can model the data by grouping it based on the covariates. We then permute the patient labels and refit the models to generate a null distribution and calculate empirical p-values.

FDR correction is performed after running all models using the Benjamini Bogomolov method. We cluster features in families using the spearman correlation of the log likelihood ratio test statistic across all permutations. The simes p-value is calculated for all families and used to select significant clusters of features. Then the BH correction is applied within each cluster and compared to the B.B. "adjusted" significance threshold based on the number of significant clusters / total number of clusters. Features with a BH adjusted p-value less than the adjusted threshold are deemed significant.

Once a feature is found to be significant, the current practice is to perform post-hoc pairwise comparisons, to see which specific comparisons are driving the significance. These need to be FDR corrected within each feature but not between different features. 

Caveat: Most of these aren't actually GLMs but since they were used in the original version the name stuck

### Scripts

1. `BetaGLM.py`, `GammaGLM.py`, `NegativeBinomial.py`, and `ZeroInflatedNegativeBinomial.py` - python scripts that use statsmodels base models to perform beta regression and gamma regression respectively

2. `statsmodels_utils.py` - script containing helper functions for the regression modeling and plotting. 

3. `GLMCollection.py` - Main script that contains the modeling, plotting, fitting, and permutation testing

4. `tutorial_feature_comparison.ipynb` - Tutorial notebook outlining the main functions of the GLMCollection and helper scripts

5. `GeneAnalyzer.py` - Class that handles differential expression and pathway enrichment analyses

6. `tutorial_gene_analysis.ipynb` - Tutorial notebook for GeneAnalyzer


### Modifications for other analyses

Most other analyses will use similar features calculated on different datasets. If the goal is to find significant differences (not to find predictors for different conditions), the code can be run with minimal modifications. The following list is mainly clerical/syntax changes

* [`get_celltype_annot_region_feature_type`](https://github.com/Tyler-Lam/Knottlab-Tutorials/blob/50f584f70403b7678448421de84a09e25525a0ce/regression_models/statsmodels_utils.py#L69])
   * This function takes a feature name and returns the cell type, annotated region, and feature type
   * You will likely have to change the `annot_dict` to match your specific annotation regions
   * Modify the function to parse your feature name conventions if they differ from mine
* [`_parse_feature_name`](https://github.com/Tyler-Lam/Knottlab-Tutorials/blob/50f584f70403b7678448421de84a09e25525a0ce/regression_models/GLMCollection.py#L157)
   * This function is used by the GLM collection to get the cell type, region, columns used for the regression models, and model type
   * If your feature names differ from mine, this will have to be changed
   * If you want to use a different model (e.g. gamma instead of negative binomial) you would change the `model_type` here to match the conventions in [`add_models_batch`](https://github.com/Tyler-Lam/Knottlab-Tutorials/blob/50f584f70403b7678448421de84a09e25525a0ce/regression_models/GLMCollection.py#L293)

### Setup environment for GeneAnalyzer

I use a modified version of PyDESeq2 that has an implementation of the likelihood ratio test. This is not present in the original PyDESeq2 code as of the time of making this code. My modified version can be found [here](https://github.com/Tyler-Lam/PyDESeq2) and is based on PyDESeq version 0.5.3

You can install this using github into a compatible conda environment using the following
```
$ git clone https://github.com/Tyler-Lam/PyDESeq2.git
$ cd PyDESeq2
$ conda activate myenv
$ pip install -e .
```

Alternatively (for my coworkers), you can copy my environment from `/common/lamt2/miniforge3/envs/pydeseq2_dev` and do
```
$ pip install -e /common/lamt2/src/PyDESeq2/
```
