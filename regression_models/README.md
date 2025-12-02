# Regression Model Analysis Pipeline

Last updated 11/7/25 - Tyler Lam

Scripts used to compare features using various regression models. These models take feature dataframes and construct GLMs/regression models to extract p-values using contrast vectors.

We have two main models that we use to compare features: Beta regression (when feature are bound between 0-1), and gamma regression (when features are bound between 0-inf). The reason we use these models instead of bernoulli or poisson is due to the overdispersion from interpatient variance. Essentially, the variance between patients is much higher than the variance from either of these distributions.

We use the beta distribution to model proportions/fractions from continuous counts. Celltype proportions are given as n_cells / n_total_cells, while the covering fractions are measured in terms of n_pixels / n_total_pixels, which are discrete counts. Since these can contain exact 0/1 values, we apply the [Smithson–Verkuilen correction](https://pubmed.ncbi.nlm.nih.gov/16594767/).

Count density (n_cells / area) and 1+spatial correlations (n_exp / n_obs) can be modeled with gamma distributions (continuous) or discrete (zero-inflated) negative binomial distributions. For the gamma distribution, there is no similar method to correct exact 0 numerators, but since the numerators are discrete counts we adjust exact 0 counts to 0.1. TBD if this is correct but in general it produces reasonable results. For more precise fitting, the discrete methods can be used but take longer to run. A method of moments estimate of fit parameters is used to calculate the predicted zero counts and compared to the observed. If there are significantly more observed zeros than expected, the feature is added using a zero-inflated negative binomial.

Once all models are fit, we perform permutation testing to calculate empirical FDR values. The initial test to extract significant features uses the log likelihood ratio with a reduced model as a test statistic. This effectively measures how much better we can model the data by grouping it based on the covariates. We then permute the patient labels and refit the models to generate a null distribution and calculate empirical p-values.

FDR correction is performed after running all models using the Benjamini Bogomolov method. We cluster features in families using the spearman correlation of the log likelihood ratio test statistic across all permutations. The simes p-value is calculated for all families and used to select significant clusters of features. Then the BH correction is applied within each cluster and compared to the B.B. "adjusted" significance threshold based on the number of significant clusters / total number of clusters. Features with a BH adjusted p-value less than the adjusted threshold are deemed significant.

Once a feature is found to be significant, the current practice is to perform post-hoc pairwise comparisons, to see which specific comparisons are driving the significance. These need to be FDR corrected within each feature but not between different features. 

### Scripts

1. `BetaGLM.py`, `GammaGLM.py`, `NegativeBinomial.py`, and `ZeroInflatedNegativeBinomial.py` - python scripts that use statsmodels base models to perform beta regression and gamma regression respectively

2. `statsmodels_utils.py` - script containing helper functions for the regression modeling and plotting. 

3. `GLMCollection.py` - Main script that contains the modeling, plotting, fitting, and permutation testing

4. `tutorial_feature_comparison.ipynb` - Tutorial notebook outlining the main functions of the GLMCollection and helper scripts