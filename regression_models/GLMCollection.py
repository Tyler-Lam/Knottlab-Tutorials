from statsmodels_utils import *
from BetaGLM import *
from GammaGLM import *
from NegativeBinomial import *
from ZeroInflatedNegativeBinomial import *

# Copied from Aagam's optimized code
def _fit_single_model(model_name, model, maxiter = 5000, fit_kwargs = {}):
    """
    Fits a single GLM model, handles convergence, and captures important warnings.
    Returns a tuple: (model_name, fit_result, convergence_status, list_of_warnings).
    """
    result = None
    converged = False
    important_warnings = []

    # Use the context manager to capture any warnings produced during the fit
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')  # Capture all warnings

        try:
            # First fit attempt
            result = model.fit(**fit_kwargs)
            converged = not np.isnan(result.bse).any() and result.mle_retvals['converged']
            
            # If not converged, try increasing the max-iterations
            if not converged:
                w.clear()
                result = model.fit(maxiter = maxiter, **fit_kwargs)
                converged = not np.isnan(result.bse).any() and result.mle_retvals['converged']
                
            # If that doesn't work, loop through different fit methods (sometimes this works?)
            methods = ['bfgs', 'lbfgs', 'nm', 'newton', 'powell']
            half_converged = [] # If we get finite coefficients but no variance/std error
            if not converged:
                if not converged and not np.isnan(result.bse).any():
                    half_converged.append('bfgs')
                for i, m in enumerate(methods[1:]):
                    w.clear()
                    result = model.fit(method = m, maxiter = maxiter, **fit_kwargs)
                    converged = not np.isnan(result.bse).any() and not np.isnan(result.bse).any() 
                    if converged:
                        w.clear()
                        break
                    if not converged and not np.isnan(result.bse).any():
                        half_converged.append(m)
                        
            # If no methods fully converged, use the first one that half-converged
            if not converged:
                if len(half_converged) > 0:
                    w.clear()
                    warnings.warn(f"Fit failed with all methods. Using {half_converged[0]} method", UserWarning)
                    result = model.fit(method = half_converged[0], maxiter = maxiter, **fit_kwargs)
                    
        except Exception as e:
            # If fitting crashes, record the error as a warning and return failure
            important_warnings.append(f"Fit failed with error: {e}\n{traceback.format_exc()}")
            return model_name, None, False, important_warnings

    # After the fit, filter the captured warnings
    for warn in w:
        msg = str(warn.message).lower()
        # Ignore common, non-critical numerical warnings
        if 'encountered in subtract' in msg or'df_resid' in msg or 'divide by zero' in msg or 'invalid value encountered' in msg or 'overflow encountered' in msg:
            continue
        important_warnings.append(warnings.formatwarning(
            warn.message,
            warn.category,
            warn.filename,
            warn.lineno,
            warn.line,)
        )

    return (model_name, result, converged, important_warnings)

def _run_single_comparison(res, c_test, c_ref, name):
    """
    Get the stat results for one model given a test contrast, ref contrast, and model name
    Used to parallelize stat calculations for all 
    
    Parameters:
    -----------
    * res: Model fit result
    * c_test, c_ref (pd.Series): Test and reference contrast vectors from GLMCollection.cond()
    * name (str): Model name
    
    Returns a tuple of (t-wald, p-wald, effect, effectSE, name)
    """
    y = res.model.endog
    x = res.model.exog
    
    idx_t = set(c_test.index.values)
    idx_t_new = [x for x in res.model.exog_names if x in idx_t]
    idx_r = set(c_ref.index.values)
    idx_r_new = [x for x in res.model.exog_names if x in idx_r]
    c_test = c_test[idx_t_new]
    c_ref = c_ref[idx_r_new]
    c = c_test - c_ref
    
    # If the result is not converged
    if np.isnan(res.bse).any() or not res.mle_retvals['converged']:
        return (np.nan, np.nan, np.nan, np.nan, name)
    # If the comparison contains categories that were removed from the given model
    if not idx_t.issubset(set(res.model.exog_names)) or not idx_r.issubset(set(res.model.exog_names)):
        return (np.nan, np.nan, np.nan, np.nan, name)
    try:
        contrast = pd.Series(0, index = res.params.index)
        contrast.loc[c.index] = c
        res_wald = res.wald_test(contrast, scalar = True)
        res_t = res.t_test(contrast)
    except:
        return (np.nan, np.nan, np.nan, np.nan, name)

    return (res_wald.statistic, res_wald.pvalue, res_t.effect.item(), res_t.sd.item(), name)
    
    
class GLMCollection():
    def __init__(
        self,
        features,
        preselection,
        group_key,
        comparisons,
        formula = None,
        null_formula = None,
        agg_features = None):
        """
        Class to store beta-binomial and gamma-poisson GLMs and perform fits/permutation testing
        
        Parameters:
        -----------
        features: pd.DataFrame
            dataframe indexed by patient/core/sample with features and metadata as columns. See tutorial for formatting
        preselection: dict | str | List[str]
            Filtering criteria for patients
            If dictionary, key: val pairs will be filtered with df[key] == val
            If str or list[str], use pd.eval() to filter by all equations
        group_key: str | List[str]
            Metadata columns that define groups for aggregating features. Generally by patient
        comparisons: str | List[str]
            Column(s) to specify covariate(s) in the design matrix
        """
        
        self.features_raw = features
        self.preselection = [preselection] if isinstance(preselection, str) else preselection
        self.group_key = group_key if isinstance(group_key, list) else [group_key]
        self.comparisons = comparisons if isinstance(group_key, list) else [comparisons]

        if isinstance(self.preselection, dict):    
            # Filter based on preselection and nan-values in predictors
            self.features_raw = self.features_raw[(self.features_raw[list(self.preselection.keys())] == pd.Series(self.preselection)).all(axis = 1)].copy()
        elif isinstance(self.preselection, list):
            self.features_raw = self.features_raw[self.features_raw.eval(' & '.join(f'({x})' for x in self.preselection))].copy()
        self.features_raw = self.features_raw[~self.features_raw[self.group_key + self.comparisons].isna().any(axis = 1)]

        # Aggregated feature dataframe
        self.agg_features = agg_features

        self.formula = "*".join(comparisons) if formula is None else formula
        # Use formulaic contrasts to get contrast vectors
        self.fc = FormulaicContrasts(self.features_raw, f'~ {self.formula}')
        self.null_formula = null_formula
        self.fc_null = FormulaicContrasts(self.features_raw, '~ 1') if self.null_formula is None else FormulaicContrasts(self.features_raw, f'~ {self.null_formula}')
        
        self.features = {}
        self.models = {}
        self.models_null = {}
        self.models_info = {}
        self.results = {}
        self.results_null = {}
        self.converged = defaultdict(lambda: False)
        self.converged_null = defaultdict(lambda: False)
        self.isfit = False
        self.isfit_null = False
        
    def _parse_feature_name(self, c, skip = defaultdict(list)):
        """
        Function to parse feature names, getting columns and model type.
        Assumes formatting of feature dataframe found in tutorial to determine if feature if Beta-binomial, beta, or gamma-Poisson and count/val/scale col names
        
        Parameters:
        --------------
        c: str
            Column name for feature
        skip: dict
            Dictionary of celltypes per category to skip due to low statistics or redundancy (eg {'primary_celltype': ['neural', 'muscle']} to skip comparisons using neural/muscle as primary_celltypes)
        """
        
        info = {'val_col': c}
        # Pattern I use for child/parent proportions
        subtype_pattern = re.compile(f".*proportion_per_.*_per_annot_region_.*")
        
        # If column is total proportion per annotated region
        if 'proportion_per_annot_region' in c:
            tier =  c.split('_proportion_per_annot_region_')[0]
            ct = c.split('___')[1]
            if ct in skip[tier]:
                return None
            counts_col = c.replace('proportion', 'counts')
            annot_region = c.split('___')[0][-1]
            totals_col = f'total_counts_per_annot_region_{annot_region}'
            info['counts_col'] = counts_col
            info['totals_col'] = totals_col
            info['model_type'] = 'Beta'
            return info
        
        # If column is child/parent proportions
        elif subtype_pattern.match(c):
            tier = c.split('_proportion')[0]
            parent_tier = {'secondary_celltype': 'primary_celltype', 'tertiary_celltype': 'secondary_celltype'}
            child_ct = c.split('___')[1]
            parent_ct = c.split('_per_')[1]
            if child_ct in skip[tier] or parent_ct in skip[parent_tier[tier]]:
                return None
            region = c.split('_annot_region_')[1][0]
            counts_col = f'{tier}_counts_per_annot_region_{region}___{child_ct}'
            totals_col = f'{parent_tier[tier]}_counts_per_annot_region_{region}___{parent_ct}'
            info['counts_col'] = counts_col
            info['totals_col'] = totals_col
            info['model_type'] = 'Beta'
            return info
                    
        # If column is density per annotated region
        elif 'density_per_annot_region' in c:
            tier =  c.split('_density_per_annot_region_')[0]
            region = c.split('_annot_region_')[1][0]
            ct = c.split('___')[1]
            if ct in skip[tier]:
                return None
            counts_col = c.replace('density', 'counts')
            annot_region = c.split('___')[0][-1]
            scale_col = f'annot{region}_area'
            info['counts_col'] = counts_col
            info['scale_col'] = scale_col
            info['model_type'] = 'NegativeBinomial'
            return info
        
        # If column is spatial correlations
        elif 'spatial_correlation' in c:
            tier = c.split('_spatial_correlation')[0]
            ct1 = c.split('_central_cell_')[1].split('_radius')[0]
            ct2 = c.split('___')[-1]
            if ct1 in skip[tier] or ct2 in skip[tier]:
                return None
            annot_region = c.split('annot_region_')[1][0]
            counts_col = c.replace("spatial_correlation", "observed_count")
            scale_col = c.replace("spatial_correlation", "expected_count")
            info['counts_col'] = counts_col
            info['scale_col'] = scale_col
            info['model_type'] = 'NegativeBinomial'
            return info
            
        # For covering fraction
        elif "covering_fraction" in c:
            tier = c.split('_covering_fraction')[0]
            ct1 = c.split('_covering_fraction_')[1].split('_radius')[0]
            ct2 = c.split('___')[-1]
            if ct1 in skip[tier] or ct2 in skip[tier]:
                return None
            region = c.split('annot_region_')[1][0]
            counts_col = c.replace("covering_fraction", "n_pixels")
            totals_col = c.replace("covering_fraction", "total_pixels")
            info['counts_col'] = counts_col
            info['totals_col'] = totals_col
            info['model_type'] = 'Beta'            
            return info
        
        else:
            print("Column does not match format for proportions, densities, or spatial correlations")
            return None
        
    def add_models_batch(self, cols_to_add, skip = defaultdict(list), verbose = True, show_progress = True):
        """
        Add all models to the GLMCollection. From Aagam's modifications
        
        Parameters:
        ------------
        * cols_to_add (List[str]): List of columns to add to model
        * skip: Dictionary of cell types to skip (see _parse_feature_name method)
        """
        
        # Get the model info for all columns to add
        all_metas = []
        for c in cols_to_add:
            meta = self._parse_feature_name(c, skip = skip)
            if meta:
                all_metas.append(meta)
        
        if not all_metas:
            print("No valid models added")
            return
        
        # If the model does not have aggregated features, do the groupings
        if self.agg_features is None:

            counts_cols = set()
            scale_cols = set()
            totals_cols = set()
            for m in all_metas:
                counts_cols.add(m['counts_col'])
                if 'scale_col' in m:
                    scale_cols.add(m['scale_col'])
                if 'totals_col' in m:
                    totals_cols.add(m['totals_col'])
            
            all_needed_cols = list(set(self.group_key + self.comparisons) | counts_cols | scale_cols | totals_cols)
            sub_df = self.features_raw[all_needed_cols]
            grouped = sub_df.groupby(self.group_key + self.comparisons, observed = True)
            self.agg_features = grouped.sum()
            
        warning_list = []

        # --- 4. Loop through metadata to create models from aggregated data ---
        for meta in tqdm(all_metas, desc="Creating Models from Batch", disable=not show_progress, total = len(all_metas), leave=True):
            with warnings.catch_warnings(record = True) as w:

                val_col = meta['val_col']
                counts_col = meta['counts_col']
                # --- A. Construct feature-specific aggregated DataFrame ---
                agg_df = pd.DataFrame(self.agg_features[counts_col]).rename(columns={counts_col: 'counts'})
                
                # Require categories to have >=2 patients with nonzero counts
                nonzero = agg_df.groupby(self.comparisons)['counts'].apply(lambda x: sum(x > 0)) >= 2
                valid = nonzero[nonzero].index.to_frame(index = False)
                if len(valid) <= 1:
                    warnings.warn(f'Category {val_col} has less than 2 valid categories. Skipping model.', UserWarning)
                    for warn in w:
                        warning_list.append((val_col, warn.message))
                    continue
                
                if len(valid) < len(nonzero):
                    warnings.warn(f"Category {val_col} removed {len(nonzero) - len(valid)} invalid groups from design matrix", UserWarning)
                    
                if meta['model_type'] == 'Beta':
                    totals_col = meta['totals_col']
                    agg_df['total_counts'] = self.agg_features[totals_col]
                    agg_df = agg_df[agg_df['total_counts'] > 0].copy()
                    agg_df = agg_df.merge(valid, on = self.comparisons, how = 'inner')
                    if agg_df.empty: 
                        warnings.warn(f'Category {val_col} has 0 nonzero total counts. Skipping model.', UserWarning)
                        for warn in w:
                            warning_list.append((val_col, warn.message))
                        continue
                    # Adjust exact 0 and 1 proportions using Smithson and Verkuilen method
                    # https://www.researchgate.net/publication/7184584_A_better_lemon_squeezer_Maximum-likelihood_regression_with_beta-distributed_dependent_variables
                    agg_df['rate'] = (agg_df['counts'] / agg_df['total_counts'] * (agg_df['total_counts'] - 1) + 0.5) / agg_df['total_counts']

                elif meta['model_type'] == 'Gamma':
                    scale_col = meta['scale_col']
                    agg_df['scale'] = self.agg_features[scale_col]
                    agg_df = agg_df[agg_df['scale'] > 0].copy()
                    agg_df = agg_df.merge(valid, on = self.comparisons, how = 'inner')
                    if agg_df.empty: 
                        warnings.warn(f'Category {val_col} has 0 nonzero total counts. Skipping model.', UserWarning)
                        for warn in w:
                            warning_list.append((val_col, warn.message))
                        continue
                    agg_df['rate'] = (agg_df['counts'] / agg_df['scale'])
                    # For 0 gamma counts, add 0.1 counts and divide by scale to get the final rate
                    # This can be weird sometimes so make sure you check values to see if they make sense
                    #    e.g. adding .1 count to a small area can give a large density
                    agg_df.loc[agg_df['counts'] == 0, 'rate'] = 0.1 / agg_df.loc[agg_df['counts'] == 0]['scale']
                    
                elif meta['model_type'] == 'NegativeBinomial' or meta['model_type'] == 'ZeroInflatedNegativeBinomial':
                    scale_col = meta['scale_col']
                    agg_df['scale'] = self.agg_features[scale_col]
                    agg_df = agg_df[agg_df['scale'] > 0].copy()
                    agg_df = agg_df.merge(valid, on = self.comparisons, how = 'inner')
                    if agg_df.empty: 
                        warnings.warn(f'Category {val_col} has 0 nonzero total counts. Skipping model.', UserWarning)
                        for warn in w:
                            warning_list.append((val_col, warn.message))
                        continue
                    agg_df['rate'] = (agg_df['counts'] / agg_df['scale'])
                    agg_df.loc[agg_df['counts'] == 0, 'rate'] = agg_df.loc[agg_df['counts'] > 0]['rate'].min()
                    
                agg_df.reset_index(inplace=True)
                agg_df['classification'] = agg_df.apply(lambda x: '___'.join([f'{c}__{x[c]}' for c in self.comparisons]), axis=1)

                self.features[val_col] = agg_df
                
                # --- B. Validate and create the model object ---

                if meta['model_type'] == 'Beta':
                    y, X = dmatrices(f'rate ~ {self.formula}', agg_df, return_type = 'dataframe')
                    if self.null_formula is None:
                        null_X = sm.add_constant(pd.DataFrame({"const": 1}, index = agg_df.index))
                    else:
                        null_Y, null_X = dmatrices(f'rate ~ {self.null_formula}', data = agg_df, return_type = 'dataframe')
                    # Edge cases: dropping exog columns with only 1 value
                    # For example if all patients are HPV+ then we remove the HPV column from the design matrix
                    for col in X.columns[1:]:
                        if X[col].nunique() <= 1:
                            X = X.drop(col, axis = 1)
                    for col in null_X.columns[1:]:
                        if null_X[col].nunique() <= 1:
                            null_X = null_X.drop(col, axis = 1)
                    
                    self.models[val_col] = BetaGLM(y, X)
                    self.models_null[val_col] = BetaGLM(y, null_X)
                    
                elif meta['model_type'] == 'Gamma':
                    y, X = dmatrices(f'rate ~ {self.formula}', agg_df, return_type = 'dataframe')
                    if self.null_formula is None:
                        null_X = sm.add_constant(pd.DataFrame({"const": 1}, index = agg_df.index))
                    else:
                        null_Y, null_X = dmatrices(f'rate ~ {self.null_formula}', data = agg_df, return_type = 'dataframe')
                    # Edge cases: dropping exog columns with only 1 value
                    for col in X.columns[1:]:
                        if X[col].nunique() <= 1:
                            X = X.drop(col, axis = 1)
                    for col in null_X.columns[1:]:
                        if null_X[col].nunique() <= 1:
                            null_X = null_X.drop(col, axis = 1)
                    
                    self.models[val_col] = GammaGLM(y, X)
                    self.models_null[val_col] = GammaGLM(y, null_X)

                elif meta['model_type'] == 'NegativeBinomial':
                    y, X = dmatrices(f'counts ~ {self.formula}', agg_df, return_type = 'dataframe')
                    offset = np.log(agg_df['scale'])
                    if self.null_formula is None:
                        null_X = sm.add_constant(pd.DataFrame({"const": 1}, index = agg_df.index))
                    else:
                        null_Y, null_X = dmatrices(f'counts ~ {self.null_formula}', data = agg_df, return_type = 'dataframe')
                    # Edge cases: dropping exog columns with only 1 value
                    for col in X.columns[1:]:
                        if X[col].nunique() <= 1:
                            X = X.drop(col, axis = 1)
                    for col in null_X.columns[1:]:
                        if null_X[col].nunique() <= 1:
                            null_X = null_X.drop(col, axis = 1)
                    
                    # If a model has more zeros than expected, use a zero-inflated model
                    # Otherwise use a standard negative binomial
                    # Predict using the method of moments estimate
                    mu = (agg_df['counts'] / agg_df['scale']).mean()
                    var = (agg_df['counts'] / agg_df['scale']).std()**2
                    alpha_MOM = (var - mu) / (mu**2) #since var = mu + alpha * mu^2
                    pct_zero_exp = (1 / (1 + alpha_MOM * mu))**(1/alpha_MOM)
                    pct_zero = sum(agg_df['counts'] == 0) / len(agg_df['counts'])
                    if (pct_zero > 1.5 * pct_zero_exp and pct_zero > pct_zero_exp + 0.05) or pct_zero > 0.1:
                        self.models[val_col] = ZeroInflatedNegativeBinomialCustom(y, X, offset = offset)
                        self.models_null[val_col] = ZeroInflatedNegativeBinomialCustom(y, null_X, offset = offset)
                    else:
                        self.models[val_col] = NegativeBinomialCustom(y, X, offset = offset)
                        self.models_null[val_col] = NegativeBinomialCustom(y, null_X, offset = offset)

                for warn in w:
                    warning_list.append((val_col, warn.message))
                        
        if verbose:
            for idx, warning in warning_list:
                print(f'   Warning adding {idx}: {warning}')

    def fit_models_parallel(self, n_jobs = None, verbose = True, show_progress = True, fit_kwargs = {}):
        """
        Fit all models in parallel
        
        Parameters:
        -----------
        
        """
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
            
        if n_jobs == 1:
            results = [_fit_single_model(name, model, fit_kwargs = fit_kwargs) for name, model in tqdm(self.models.items(), desc = "Fitting full models", disable = not show_progress)]
        
        else:
            tasks = [
                delayed(_fit_single_model)(name, model, fit_kwargs = fit_kwargs)
                for name, model in self.models.items()
            ]
            
            with tqdm_joblib(tqdm(desc = "Fitting full models", total = len(self.models), disable = not show_progress)) as pbar:
                results = Parallel(n_jobs = n_jobs)(tasks)
                
        warning_list = []
        for model_name, result, converged, warnings_from_worker in results:
            if result is not None:
                self.results[model_name] = result
            self.converged[model_name] = converged
            # Collect warnings
            if warnings_from_worker:
                for msg in warnings_from_worker:
                    warning_list.append((model_name, msg))
        if verbose:
            for idx, warning in warning_list:
                print(f'   warning when fitting {idx}: {warning}')
        self.isfit = True

    def fit_null_models_parallel(self, n_jobs = None, verbose = True, show_progress = True, fit_kwargs = {}):
        
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
            
        if n_jobs == 1:
            results = [_fit_single_model(name, model, fit_kwargs = fit_kwargs) for name, model in tqdm(self.models_null.items(), desc = "Fitting full models", disable = not show_progress)]
        else:
            tasks = [
                delayed(_fit_single_model)(name, model, fit_kwargs = fit_kwargs)
                for name, model in self.models_null.items()
            ]
            
            with tqdm_joblib(tqdm(desc = "Fitting null models", total = len(self.models_null), disable = not show_progress)) as pbar:
                results = Parallel(n_jobs = n_jobs)(tasks)
                
        warning_list = []
        for model_name, result, converged, warnings_from_worker in results:
            if result is not None:
                self.results_null[model_name] = result
            self.converged_null[model_name] = converged
            # Collect warnings
            if warnings_from_worker:
                for msg in warnings_from_worker:
                    warning_list.append((model_name, msg))
        if verbose:
            for idx, warning in warning_list:
                print(f'   warning when fitting {idx}: {warning}')
        self.isfit_null = True

    def resid(self, **kwargs):
        """
        Calculate the residuals of the fits and return one array containing all residuals
        
        Parameters:
        -----------
        * kwargs: kwargs when calculating residuals. Generally "which" = ['linear', 'rate', or 'pearson'] to specify the type of residual to calculate
        """
        out = np.array([])
        for model in self.models:
            if model in self.results and self.converged[model]:
                results = self.results[model]
            else:
                continue
            out = np.append(out, results.resid(**kwargs))
        return out

    def cond(self, model = None, **kwargs):
        """
        Get the contrast vector using Formulaic Contrasts. To get difference between two classes, you subtract the contrast vectors and take the dot product with the fit coefficients
        These are formatted as pandas series, where the row index is the design matrix col name and the value is 0 or 1
        
        Parameters:
        ------------
        kwargs: parameter = values must are predictors and categories to get the contrast for specific data classes
        
        Example: My design matrix used the formula "counts + total_counts ~ HR_HPV + P53 + HR_HPV:P53
        To get the contrast comparing P53+ HPV- (test) with P53- HPV- (ref), I would do:
        c_test = self.cond(P53 = 1, HR_HPV = 0)
        c_ref = self.cond(P53 = 0, HR_HPV = 0)
        c = c_test - c_ref
        """
        
        contrast = self.fc.cond(**kwargs)
        if model is not None and model in self.models:
            try:
                with warnings.catch_warnings(record = True) as w:
                    warnings.simplefilter('always')
                    fc = FormulaicContrasts(self.features[model], f'~ {self.formula}')
                    contrast = fc.cond(**kwargs)
            except:
                pass
        return contrast
    
    def cond_null(self, model = None, **kwargs):
        """
        Get the contrast vector using Formulaic Contrasts. To get difference between two classes, you subtract the contrast vectors and take the dot product with the fit coefficients
        These are formatted as pandas series, where the row index is the design matrix col name and the value is 0 or 1
        
        Parameters:
        ------------
        kwargs: parameter = values must are predictors and categories to get the contrast for specific data classes
        
        Example: My design matrix used the formula "counts + total_counts ~ HR_HPV + P53 + HR_HPV:P53
        To get the contrast comparing P53+ HPV- (test) with P53- HPV- (ref), I would do:
        c_test = self.cond(P53 = 1, HR_HPV = 0)
        c_ref = self.cond(P53 = 0, HR_HPV = 0)
        c = c_test - c_ref
        """
        
        contrast = self.fc_null.cond(**kwargs)
        if model is not None and model in self.models_null:
            try:
                with warnings.catch_warnings(record = True) as w:
                    warnings.simplefilter('always')
                    fc = FormulaicContrasts(self.features[model], f'~ {self.null_formula}')
                    contrast = fc.cond(**kwargs)
            except:
                pass
        return contrast

    def hist_plot(self, model, fitted = False, order = None, logx = False, logy = False, ax_label = '', figsize = (8, 5), show = False, ax = None, title = ''):
        """
        Plot a histogram of the distribution of data for a given model. Top plot is a normalized histogram for each feature category (unstacked). Bottom plot is a strip + violin plot with points colored by the denominator (total_counts or scale)
        
        Parameters:
        ------------
        * model (str): Model name to plot
        * fitted: Plot the fitted model. We interpret the beta-binomial and gamma-poisson as mixed models on the means, and use the underlying beta or gamma distributions as the pdf. These are overlaid on the histograms and used as the envelope for the violin plots
        * order: Order (and subset) of classifications to plot. These must be formatted the same way as the feature "classification" column made when getting the aggregated features
                 If I am comparing HR_HPV and Diagnosis stage but only want LSIL and HSIL for HPV+ in a specific order, I would use order = ['HR_HPV__1.0___Diagnosis__No_SIL', 'HR_HPV__1.0___Diagnosis__LSIL']
        * logy: Use a log scale for the y-axis
        * ax_label: Axis label for the feature. This is the x-axis on the histogram and y-axis on the violin plots
        """
        if fitted:
            if model not in self.results.keys() or not self.converged[model]:
                print (f'Model {model} does not have a fitted model')
                fitted = False
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize, constrained_layout = True)
        x_pos = {cat: i for i, cat in enumerate(self.features[model]['classification'].unique())}
        if order is not None:
            if len(set(order).intersection(set(self.features[model]['classification'].unique()))) == 0:
                print(f"Given order has no values in the model's classification categories.\nPossible values are: {', '.join(self.features[model]['classification'].unique())}")
            else:
                x_pos = {cat: i for i, cat in enumerate(order)}
                
        # Get uniform binning for all histograms
        bins = np.histogram_bin_edges(self.features[model]['rate'], bins = 'auto')
        x = np.linspace(bins[0], bins[-1], 1000)
        if logx:
            bins = np.logspace(np.log10(bins[0] if bins[0] > 0 else self.features[model]['rate'].min()), np.log10(bins[-1]), 10)
            x = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 1000)
            
        # Make the histogram
        sns.histplot(
            x = 'rate',
            hue = 'classification', 
            data = self.features[model],
            bins = bins, 
            stat = 'density', 
            alpha = 0.4, 
            hue_order = order,
            common_norm = False, 
            ax = ax,
        )
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ymin, ymax = ax.get_ylim()
        
        # If using the fitted model, get the pdf for each classification
        if fitted:
            pdfs = {}
            pdf_max = 0 # Max value of pdf for shared normalization
            for i, cat in enumerate(x_pos.keys()):
                if cat not in self.features[model]['classification'].unique():
                    continue
                # Get col: val pairs from the classification names
                vals ={x.split('__')[0]: self.features[model][x.split('__')[0]].dtype.type(x.split('__')[1]) for x in cat.split('___')}
                # Use col: val pairs to get the given contrast
                contrast = self.cond(model = model, **vals)
                # Sometimes the contrast orders don't match. Fix that here:
                result = self.results[model]
                # For the get_pdf methods, contrast is only for the regressor coefficients
                pdfs[cat] = result.get_pdf(x, contrast)
                pdf_max = max(pdf_max, pdfs[cat][np.isfinite(pdfs[cat])].max())
                ax.plot(x, pdfs[cat], '-')
                
            ax.set_ylim(None, ymax if (ymax > pdf_max) else min(1.2 * pdf_max, 2*ymax))
        # Histogram plot formatting
        ax.set_ylabel("a.u.")
        ax.set_xlabel(ax_label)  
        ax.set_title(title)
        ax.grid()
        
        if show:
            plt.show()
        else:
            return ax

    def strip_plot(self, model, fitted = False, order = None, log = False, cbar_label = '', ax_label = '', figsize = (8, 5), show = False, save_dir = None, ax = None, title = ''):
        """
        Plot a strip plot and envelope of the distribution of data for a given model. Top plot is a normalized histogram for each feature category (unstacked). Bottom plot is a strip + violin plot with points colored by the denominator (total_counts or scale)
        
        Parameters:
        ------------
        * model (str): Model name to plot
        * fitted: Plot the fitted model. We interpret the beta-binomial and gamma-poisson as mixed models on the means, and use the underlying beta or gamma distributions as the pdf. These are overlaid on the histograms and used as the envelope for the violin plots
        * order: Order (and subset) of classifications to plot. These must be formatted the same way as the feature "classification" column made when getting the aggregated features
                 If I am comparing HR_HPV and Diagnosis stage but only want LSIL and HSIL for HPV+ in a specific order, I would use order = ['HR_HPV__1.0___Diagnosis__No_SIL', 'HR_HPV__1.0___Diagnosis__LSIL']
        * logy: Use a log scale for the y-axis
        * cbar_label: Label for the colorbar. This always uses the denominator when calculating the feature, so for proportions the label should be "Total cells", densities should be "Area", etc.
        * ax_label: Axis label for the feature. This is the x-axis on the histogram and y-axis on the violin plots
        """
        if fitted:
            if model not in self.results.keys() or not self.converged[model]:
                print (f'Model {model} does not have a fitted model')
                fitted = False
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize, constrained_layout = True)
        x_pos = {cat: i for i, cat in enumerate(self.features[model]['classification'].unique())}
        if order is not None:
            if len(set(order).intersection(set(self.features[model]['classification'].unique()))) == 0:
                print(f"Given order has no values in the model's classification categories.\nPossible values are: {', '.join(self.features[model]['classification'].unique())}")
            else:
                x_pos = {cat: i for i, cat in enumerate(order)}
        
        # Get uniform binning for all histograms
        content, bins = np.histogram(self.features[model]['rate'], bins = 'auto', density = True)
        x = np.linspace(bins[0], bins[-1], 1000)
        if log:
            bins = np.logspace(np.log10(bins[0] if bins[0] > 0 else self.features[model]['rate'].min()), np.log10(bins[-1]), 10)
            content, bins = np.histogram(self.features[model]['rate'], bins = bins)
            x = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 1000)
        ymax = max(content) * 1.1
        
        # If using the fitted model, get the pdf for each classification
        if fitted:
            pdfs = {}
            pdf_max = 0 # Max value of pdf for shared normalization
            for i, cat in enumerate(x_pos.keys()):
                if cat not in self.features[model]['classification'].unique():
                    continue
                # Get col: val pairs from the classification names
                vals ={x.split('__')[0]: self.features[model][x.split('__')[0]].dtype.type(x.split('__')[1]) for x in cat.split('___')}
                # Use col: val pairs to get the given contrast
                contrast = self.cond(model = model, **vals)
                # Sometimes the contrast orders don't match. Fix that here:
                result = self.results[model]
                # For the get_pdf methods, contrast is only for the regressor coefficients
                pdfs[cat] = result.get_pdf(x, contrast)
                pdf_max = max(pdf_max, pdfs[cat][np.isfinite(pdfs[cat])].max())
                
        denom = 'total_counts' if isinstance(self.models[model], BetaGLM) else 'scale'
        # If not using fitted model, use a violin plot
        if not fitted:
            sns.violinplot(
                x = 'classification',
                y = 'rate',
                data = self.features[model],
                ax = ax,
                order = order,
                alpha = 0.5,
                legend = False,
                inner = None,
            )
        # Stripplot the data over the violin plot
        sns.stripplot(
            x = 'classification', 
            y = 'rate', 
            hue = denom, 
            data = self.features[model], 
            ax = ax,
            order = order,
            linewidth = 0.5,
            palette = 'flare',
            legend = False,
            size = 2.75)
        # Axis tick and colorbar formatting
        ax.tick_params(axis = 'x', labelrotation = 90)
        norm = plt.Normalize(self.features[model][denom].min(), self.features[model][denom].max())
        scalar = plt.cm.ScalarMappable(cmap = 'flare', norm = norm)
        scalar.set_array([])
        cbar = ax.figure.colorbar(scalar, ax = ax, pad = 0.02)
        cbar.set_label(cbar_label)
    
        # Use the (shared)-normalized pdfs as the envelope if using a fitted model
        if fitted:
            for i, cat in enumerate(x_pos.keys()):
                if cat not in self.features[model]['classification'].unique():
                    continue
                if not log:
                    pdf = np.clip(pdfs[cat], a_min = None, a_max = 10 * ymax) / min(10 * ymax, pdf_max)
                else:
                    pdf = pdfs[cat] / pdf_max
                ax.fill_betweenx(
                    x, 
                    x_pos[cat] - 0.45 * pdf, 
                    x_pos[cat] + 0.45 * pdf, 
                    alpha = 0.3, 
                    color = f'C{i}'
                )
        if log:
            ax.set_yscale('log')
        ax.set_ylabel(ax_label)
        ax.grid()
        ax.set_title(title)
        if show:
            plt.show()
        else:
            return ax
    
    def plot_model(self, model, fitted = False, order = None, logx = False, logy = False, cbar_label = '', ax_label = '', title = '', figsize = (8, 10), show = True, save_dir = None):
        """
        Plot the distribution of data for a given model. Top plot is a normalized histogram for each feature category (unstacked). Bottom plot is a strip + violin plot with points colored by the denominator (total_counts or scale)
        
        Parameters:
        ------------
        * model (str): Model name to plot
        * fitted: Plot the fitted model. We interpret the beta-binomial and gamma-poisson as mixed models on the means, and use the underlying beta or gamma distributions as the pdf. These are overlaid on the histograms and used as the envelope for the violin plots
        * order: Order (and subset) of classifications to plot. These must be formatted the same way as the feature "classification" column made when getting the aggregated features
                 If I am comparing HR_HPV and Diagnosis stage but only want LSIL and HSIL for HPV+ in a specific order, I would use order = ['HR_HPV__1.0___Diagnosis__No_SIL', 'HR_HPV__1.0___Diagnosis__LSIL']
        * logy: Use a log scale for the y-axis
        * cbar_label: Label for the colorbar. This always uses the denominator when calculating the feature, so for proportions the label should be "Total cells", densities should be "Area", etc.
        * ax_label: Axis label for the feature. This is the x-axis on the histogram and y-axis on the violin plots
        """
        fig, ax = plt.subplots(2, 1, figsize = figsize, constrained_layout = True)
        self.hist_plot(model, fitted = fitted, order = order, logx = logx, logy = logy, ax_label = ax_label, show = False, ax = ax[0])
        self.strip_plot(model, fitted = fitted, order = order, log = logy, cbar_label=cbar_label, ax_label=ax_label, show = False, ax = ax[1])
        fig.suptitle(title)
        if show:
            plt.show()
        else:
            return ax
            
    def get_stat_df_parallel(self, c_test, c_ref, name, n_jobs = None):
        """
        Get the stat dataframe for all models
        
        Parameters:
        -----------
        * c_test, c_ref (dict): Dictionaries defining contrast vectors for test and ref classes
        * name: Name of the current comparison
        """

        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
            
        if n_jobs == 1:
            results = [_run_single_comparison(res, self.cond(model = feat, **c_test), self.cond(model = feat, **c_ref), feat) for feat, res in self.results.items()]
        else:
            tasks = [delayed(_run_single_comparison)(res, self.cond(model = feat, **c_test), self.cond(model = feat, **c_ref), feat) for feat, res in self.results.items()]
            results = Parallel(n_jobs = n_jobs)(tasks)
            
        out_index = []
        t_wald = []
        p_wald = []
        effect = []
        effectSE = []
        
        for t, p, eff, effSE, idx in results:
            out_index.append(idx)
            t_wald.append(t)
            p_wald.append(p)
            effect.append(eff)
            effectSE.append(effSE)
            
        out = pd.DataFrame({'t-wald': t_wald, 'p-wald': p_wald, 'effect': effect, 'effectSE': effectSE}, index = out_index)
        out[['Celltype', 'Region', 'feature_type']] = out.index.to_series().apply(get_celltype_annot_region_feature_type).apply(pd.Series)
        out.index.name = 'feature'
        return {'df': out, 'c_test': c_test, 'c_ref': c_ref, 'name': name}
        
    def run_stats_with_contrasts(self, contrasts, show_progress = True, n_jobs = None):
        """
        Takes a list of contrasts to make comparisons.
        
        Parameters:
        -----------
        * contrasts (List[dict]): List of contrast info. {'test': test_contrast, 'ref': ref_contrast, 'name': name}. Contrasts must be dictionaries defining the classes
        """     
        out = defaultdict(dict)
        if not isinstance(contrasts, list):
            contrasts = [contrasts]
        for c in tqdm(contrasts, desc = 'Getting stat dataframes', disable = not show_progress):
            out[c['name']] = self.get_stat_df_parallel(c['test'], c['ref'], c['test'], n_jobs = n_jobs)   
            out[c['name']]['ref_group'] = c['ref']
            out[c['name']]['test_group'] = c['test']
        return out

    def run_stats_pairwise(self, **kwargs):
        """
        Loops over every pairwise comparisons where all predictors but 1 are fixed at the same value. Returns a dictionary where the key indicates the comparison, and the values are a stat result dictionary
        """
        
        contrasts = []
        # Define the metadata dictionary
        metadata = defaultdict(dict)
        # Loop over all predictors (comparisons)
        for i, comp in enumerate(self.comparisons):
            
            # Separate the comparisons into the current one (comp) and all others
            others = self.comparisons[0:i] + self.comparisons[i+1: len(self.comparisons)]
            
            # Get all pairwise comparisons for the given current predictor
            pairs = list(combinations(self.features_raw[comp].dropna().unique(), 2))
            
            # If there are no "other" predictors, then loop over all pairs and make the dataframe
            if len(others) == 0:
                for pair in pairs:
                    name = f'{comp}__{pair[0]}_vs_{pair[1]}'
                    cont_test = {comp: pair[0]}
                    cont_ref = {comp: pair[1]}

                    contrasts.append({'name': name, 'test': cont_test, 'ref': cont_ref})
                    metadata[name]['comparison'] = comp
                    # Dictionaries for contrasts
                    metadata[name]['test_group'] = cont_test
                    metadata[name]['ref_group'] = cont_ref
                    # Dictionary of control variables
                    metadata[name]['control'] = self.preselection
                    
            # If there are other predictors, loop over all combinations and run pairwise comparisons for each
            else:
                for idx, row in self.features_raw[others].drop_duplicates().dropna().iterrows():
                    for pair in pairs:
                        name = f'{comp}__{pair[0]}_vs_{pair[1]}'
                        cont_test = {comp: pair[0]}
                        cont_ref = {comp: pair[1]}
                        
                        # "Other" predictors must be added to the contrast vector to get the correct coefficients
                        for c in others:
                            cont_test[c] = row[c]
                            cont_ref[c] = row[c]
                            name += f'___{c}__{row[c]}'
                        
                        contrasts.append({'name': name, 'test': cont_test, 'ref': cont_ref})
                        metadata[name] = {}
                        metadata[name]['comparison'] = comp
                        # Dictionaries for contrasts
                        metadata[name]['test_group'] = cont_test
                        metadata[name]['ref_group'] = cont_ref
                        # Dictionary of control variables
                        metadata[name]['control'] = self.preselection
                        if isinstance(self.preselection, dict):
                            metadata[name]['control'].update({c: row[c] for c in others})
                        elif isinstance(self.preselection, list):
                            metadata[name]['control'].extend([f'{c} == {row[c]}' for c in others])
        stat_res = self.run_stats_with_contrasts(contrasts, **kwargs)
        for key in stat_res:
            stat_res[key].update(metadata[key])
        return stat_res
    
    def run_stats_ref(self, ref_class, **kwargs):
        """
        Function to get all pairwise comparisons with a fixed reference group
        
        Parameters:
        -----------
        * ref_class (dict): {predictor: value} pairs that define the reference class
        """
        
        # Metadata dictionary
        metadata = defaultdict(dict)
        # Check to make sure ref_class is correctly defined
        for c in self.comparisons:
            if c not in ref_class:
                print(f"Reference class must contain value for {c}. Defaulting to pairwise comparisons")
                return self.run_stats_pairwise()

        contrasts = []
        
        # Get the contrast vector for the given reference class
        
        # Mask the reference class rows from the feature dataframe so we can loop over all other unique combinations of predictors
        mask = ~(self.features_raw[list(ref_class)] == pd.Series(ref_class)).all(axis = 1)
        for idx, row in self.features_raw[mask][self.comparisons].drop_duplicates().dropna().iterrows():
            test_class = {c: row[c] for c in self.comparisons}
            
            name = '___'.join([f"{c}__{row[c]}" for c in self.comparisons])
            contrasts.append({'name': name, 'test': test_class, 'ref': ref_class})
            metadata[name]['ref_group'] = ref_class
            metadata[name]['test_group'] = test_class
        stat_res = self.run_stats_with_contrasts(contrasts, **kwargs)
        for key in stat_res:
            stat_res[key].update(metadata[key])
        return stat_res
    
    def run_stats(self, contrasts = None, ref_class = None, **kwargs):
        """
        Wrapper function to do pairwise comparisons if no reference class or list of contrasts is specified
        
        Parameters:
        ------------
        * contrasts (List[dict]): List of contrast dictionaries
        * ref_class (dict): Dictionary specifying the reference class
        """
        if contrasts is None:
            if ref_class is None:
                # If no contrasts or ref_class given, run all pairwise comparisons
                return self.run_stats_pairwise(**kwargs)
            else:
                # If ref class given, only run comparisons relative to ref
                return self.run_stats_ref(ref_class, **kwargs)
        else:
            # If contrast given, do comparisons given contrasts
            return self.run_stats_with_contrasts(contrasts, **kwargs)
        
    def run_LLR_test(self):
        """
        Return log likelihood-ratio test for full model. This is a pseudo-anova analysis to see if there are any significant differences from the null model (with no categories)
        """

        stats = []
        indices = []
        dfs = []
        for idx in self.results:
            if not self.converged[idx] or not self.converged_null[idx]:
                continue
            lr_full = self.results[idx].llf
            lr_null = self.results_null[idx].llf
            df = self.results[idx].df_model - self.results_null[idx].df_model
            stat = max(0, 2 * (lr_full - lr_null)) # Sometimes the LLR is negative due to numerical approximations
            stats.append(stat)
            indices.append(idx)
            dfs.append(df)
        out = pd.DataFrame({'stat': stats, 'df': dfs}, index = indices)
        out.index.name = 'feature'
        out[['Celltype', 'Region', 'feature_type']] = out.index.to_series().apply(get_celltype_annot_region_feature_type).apply(pd.Series)
        return out
    
    def _shuffle_agg_df(self, df, perm_cols = None, random_state = None):
        """
        Shuffle the aggregated feature dataframe and return the shuffled dataframe
        
        Parameters:
        ------------
        * df: Pandas dataframe of all aggregated features and metadata
        * perm_cols: Columns to permute (if none, permute all patients)
        * random_state: for reproducibility when shuffling (needed to calculate correlation between features)
        """
        warnings.filterwarnings('ignore', message = '^DataFrameGroupBy.apply operated on the grouping columns.*', category = FutureWarning)

        if perm_cols is None:
            perm_cols = self.group_key + self.comparisons
        elif not isinstance(perm_cols, list):
            perm_cols = [perm_cols]
        if not set(self.group_key).issubset(set(perm_cols)):
            perm_cols = list(set(perm_cols + self.group_key))
        shuffled_df = df[perm_cols].drop_duplicates()
        shuffled_df['tmp_idx'] = shuffled_df.apply(lambda x: '_'.join(str(x[c]) for c in perm_cols), axis = 1)
        
        # Make a shuffle dictionary to map
        rng = np.random.default_rng(random_state)
        shuffle_dict = dict(zip(shuffled_df['tmp_idx'], rng.permutation(shuffled_df['tmp_idx'])))
        shuffled_df['tmp_idx'] = shuffled_df['tmp_idx'].map(shuffle_dict)

        # Re-merge the shuffled metadata and drop the temporary index
        df['tmp_idx'] = df.apply(lambda x: '_'.join(str(x[c]) for c in perm_cols), axis = 1)
        df = df.drop(columns = perm_cols)
        
        df = pd.merge(shuffled_df, df, on = 'tmp_idx', how = 'right')
        df = df.drop(columns = ['tmp_idx'])
        
        return df
        
    def _run_single_permutation(self, contrasts = None, n_jobs = None, group_cols = None, perm_cols = None, random_state = None, verbose = False):
        """
        Run the statistical analysis on a single permutation. Shuffle the patients by group_key and get the stat dataframes for each comparison
        
        Parameters:
        ------------
        * contrasts (List[dict]): List of dictionaries with contrast info. Contrasts must have 3 keys: 'name' to match the name of the comparison from the actual data stat res, 'test' and 'ref': contrasts for the test and ref groups (as pd.Series)
        * n_jobs: Number of jobs for single permutation (inner parallelization)
        * group_cols, perm_cols: Columns to group and to shuffle.
                                 Example: If I want to shuffle HPV status within diagnosis stages, I would use group_cols = 'Diagnosis' and perm_cols = 'HR_HPV'
        """
        
        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            if perm_cols is None:
                perm_cols = self.group_key + self.comparisons
            else:
                if not isinstance(perm_cols, list):
                    perm_cols = [perm_cols]
                perm_cols = perm_cols + self.group_key

            # Permute the metadata by patient + comparison
            # We include the comparison because sometimes a single patient can be in different categories
            # across multiple samples
            # e.g. one patient can have an HPV+ and HPV- sample
            df = self.agg_features.copy().reset_index()
            if not group_cols:
                df = self._shuffle_agg_df(df, random_state = random_state)
            else:
                df = df.groupby(group_cols, group_keys = False).apply(lambda g: self._shuffle_agg_df(g, perm_cols = perm_cols, random_state = random_state), include_groups = True)

            try:
                # Make a new dataframe with the shuffled features, including the aggregated features to skip the groupby function
                glms = GLMCollection(df, {}, self.group_key, self.comparisons, formula = self.formula, null_formula = self.null_formula, agg_features=df.set_index(self.group_key + self.comparisons))
                glms.add_models_batch(list(self.features.keys()), verbose = False, show_progress = False)
                glms.fit_models_parallel(verbose = False, show_progress = False, n_jobs = n_jobs)
                glms.fit_null_models_parallel(verbose = False, show_progress = False, n_jobs = n_jobs)
                
                out_res = {}

                stat_res = glms.run_stats(contrasts = contrasts, show_progress = False, n_jobs = n_jobs)
                for idx in stat_res.keys():
                    out_res[idx] = stat_res[idx]['df']
                    # Keep only important columns from the stat df
                    out_res[idx] = out_res[idx][[c for c in out_res[idx].columns if c.startswith('t-') or c in ['Celltype', 'Region', 'feature_type', 'df']]]
                # Run the pseudo-anova log-likelihood test
                out_res['llr'] = glms.run_LLR_test()
            
            except Exception as e:
                print(f"   Permutation fitting failed: {e}")
                traceback.print_exc()
                return None
        
        if verbose:
            for warn in w:
                msg = str(warn.message).lower()
                # Filter common warnings that clutter the output
                if 'stopped while some jobs were given to the executor' in msg or 'encountered in subtract' in msg or'df_resid' in msg or 'divide by zero' in msg or 'invalid value encountered' in msg or 'overflow encountered' in msg:
                    continue
                print(warnings.formatwarning(
                    warn.message,
                    warn.category,
                    warn.filename,
                    warn.lineno,
                    warn.line,
                ))
        return out_res
    
    def run_permutations_parallel(self, contrasts = None, n_permutations = 1000, show_progress = True, n_jobs = None, n_jobs_inner = None, random_state = None, **kwargs):
        """
        Run permutations in parallel
        
        Parameters:
        ------------
        * contrasts: List of dictionaries containing 'name'= name of comparison, 'c_test'= test contrast, and 'c_ref'= ref contrast. If None, default to all pairwise comparisons
        * n_permutations: Number of permutations to run
        * n_jobs: Number of jobs to run for permutation tests
        * n_jobs_inner: Number of jobs to run within each permutation test. Since we can't do nested parallelism, either this or n_jobs must be set to 1
        * random_state: seed for reproducibility
        * kwargs: arguments for _run_single_permutation()
        """
        out = defaultdict(pd.DataFrame)

        with warnings.catch_warnings(record = True) as w:
            warnings.simplefilter("always")
            # If permutations are done in series, do parallelism within each iteration
            if n_jobs == 1:
                if n_jobs_inner is None:
                    n_jobs_inner = multiprocessing.cpu_count() - 1
                results = []
                for i in tqdm(range(n_permutations), desc = 'Running permutation tests', disable = not show_progress):
                    rs = random_state if random_state is None else random_state + i
                    results.append(self._run_single_permutation(contrasts = contrasts, n_jobs = n_jobs_inner, random_state = rs, **kwargs))
            
            # If permutations are done in parallel, make each permutation take one job
            else:
                tasks = []
                for i in range(n_permutations):
                    rs = random_state if random_state is None else random_state + i
                    tasks.append(delayed(self._run_single_permutation)(contrasts = contrasts, n_jobs = 1, random_state = rs, **kwargs))
                #tasks = [delayed(self._run_single_permutation)(contrasts = contrasts, n_jobs = 1, **kwargs) for i in range(n_permutations)]
                with tqdm_joblib(tqdm(desc = "Calculating permutations", total = len(tasks), disable = not show_progress)) as pbar:
                    results = Parallel(n_jobs = n_jobs)(tasks)
                        
        for warn in w:
            msg = str(warn.message).lower()
            if 'stopped while some jobs were given to the executor' in msg:
                continue
            print(warnings.formatwarning(
                warn.message,
                warn.category,
                warn.filename,
                warn.lineno,
                warn.line,
            ))         
                
        # Loop over the results and make the output dictionary of dataframes
        for n, permutation_res in enumerate(results):
            if permutation_res is None:
                continue
            for key in permutation_res:
                tmp_df = permutation_res[key].copy()
                # Track which permutation the results are from
                # Works since joblib parallel preserves the order of tasks. No need to track random state within each task
                tmp_df['perm_iter'] = n 
                out[key] = pd.concat([out[key], tmp_df], axis = 0, join = 'outer')
        return out
    
    def run_stats_with_permutations(self, n_permutations = 1000, n_jobs = None, n_jobs_inner = None, show_progress = True, ref_class = None, contrasts = None, do_pairwise = True, **kwargs):
        """
        Run the statistical test with permutations for empirical p-values
        Empirical p-values are calculated as the fraction of null test statistics greater than the nominal wald test statistic. These are saved as p-wald-nom (or p-nom for the LLR test)
        
        (UPDATE: We no longer do this part. See the tutorial for Benjamini-Bogomolov FDR correction) ~~FDR correction is analogous to the Storey-Tribshirani multiple-trials correction, and what GSEA uses for phenotype permutation tests
        https://pubmed.ncbi.nlm.nih.gov/12883005/~~
        
        Parameters:
        ------------
        * n_permutations, n_jobs, verbose are the same as the run_permutations() function
        * n_jobs, n_jobs_inner: number of jobs for inner/outer parallelization. Due to joblib we can either do inner or outer parallelization, but not both
        * contrasts (List[dict] / dict): Contrast metadata dictionary defining the groups for pairwise comparisons
        * do_pairwise: If contrasts is None, do pairwise comparisons. If False and contrasts is None, only perform the LLR test and skip pairwise comparisons
        * ref_class (dict): Dictionary specifying the reference class predictors. If none, do all pairwise comparisons
        * kwargs: Keyword arguments for _run_single_permutation()
        """

        # Setup output dictionary
        stat_res = {}
        
        # Run comparisons if no contrasts provided
        if contrasts is None:
            contrasts = []
            if do_pairwise:
                # Get initial stat results
                stat_res = self.run_stats(ref_class = ref_class)
                
                # Get all contrasts from the original stat results
                for key in stat_res:
                    contrasts.append(dict(name = key, test = stat_res[key]['test_group'], ref = stat_res[key]['ref_group']))
        # Run comparisons if contrasts are given
        else:
            if not isinstance(contrasts, list):
                contrasts = [contrasts]
            stat_res = self.run_stats(contrasts=contrasts)
            
        # Run permutations
        
        # Setup job distribution
        # Default to outer parallelization
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
        if n_jobs_inner is None:
            n_jobs_inner = (multiprocessing.cpu_count() - 1) // n_jobs
        n_jobs_inner = min((multiprocessing.cpu_count() - 1) // n_jobs, n_jobs_inner)
        
        perm_df = self.run_permutations_parallel(contrasts, n_permutations=n_permutations, show_progress=show_progress, n_jobs = n_jobs, n_jobs_inner = n_jobs_inner, **kwargs)
        warnings.filterwarnings('ignore', message = '^DataFrameGroupBy.apply operated on the grouping columns.*', category = FutureWarning)

        # Use permutations as null distribution and apply to dataframe
        for key in stat_res:
            stat_df = stat_res[key]['df']
            
            stat_df = get_empirical_pvalues(stat_df.copy(), perm_df[key], stat_col = 't-wald')
            stat_res[key]['df'] = stat_df
            stat_res[key]['perm_df'] = perm_df[key]
        
        # Calculate permutations from the LLR test for feature-wide significance
        llr_df = self.run_LLR_test()
        llr_df = get_empirical_pvalues(llr_df.copy(), perm_df['llr'], stat_col = 'stat')
        stat_res['llr'] = {'df': llr_df, 'perm_df': perm_df['llr']}
        return stat_res

    def test_run(self, n_perm = 1000, random_state = 42):
        """
        Test to see if inner or outer parallelism would be faster for the given number of permutations
        Inner parallelism = parallelism within each permutation, permutations run in series
        Outer parallelism = each permutation run in parallel
        
        Returns dictionary of recommended n_jobs and n_jobs_inner values
        """
        print("Running test permutations to job allocation")
        print("   Running test with inner parallelism", end = ' ... ')
        t0 = time.time()
        test = self.run_permutations_parallel(n_permutations = 1, n_jobs = 1, n_jobs_inner = multiprocessing.cpu_count() - 1, show_progress=False, random_state = random_state)
        t1 = time.time()
        
        t_inner_iter = t1 - t0
        print(f"done: {t_inner_iter:.2f} s")
        print("   Running test with outer parallelism", end = " ... ")
        t0 = time.time()
        test = self.run_permutations_parallel(n_permutations = multiprocessing.cpu_count() - 1, n_jobs = multiprocessing.cpu_count() - 1, n_jobs_inner = 1, show_progress=False, random_state = random_state)
        t1 = time.time()

        t_outer_iter = t1 - t0
        print(f"done: {t_outer_iter:.2f} s")
        
        t_inner_total = t_inner_iter * n_perm
        
        t_outer_total = math.ceil(n_perm / (multiprocessing.cpu_count() - 1)) * t_outer_iter
        
        print(f'Estimated time with inner parallelism: {t_inner_total/60:.2f} min')
        print(f'Estimated time with outer parallelism: {t_outer_total/60:.2f} min')
        
        if t_inner_total < t_outer_total:
            return dict(n_jobs = 1, n_jobs_inner = multiprocessing.cpu_count() - 1)
        else:
            return dict(n_jobs = multiprocessing.cpu_count() - 1, n_jobs_inner = 1)