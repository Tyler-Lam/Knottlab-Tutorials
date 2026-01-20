from statsmodels_utils import *

class GeneAnalyzer():
    def __init__(
        self,
        adata,
        pseudobulk_key,
        group_key,
        comparisons,
        metadata = None,
        meta_merge_key = 'batch',
        is_grouped = False,
        preselection = [],
        skip = defaultdict(list),
        design = None,
        design_null = None,
        gene_set = None,
        layer = None
    ):
        """
        Class to do Deseq2 and GSVA analysis
        
        Parameters:
        ------------
        adata: AnnData
            Anndata containing gene expression matrix X, 'counts' layer
            if not using the metadata parameter, all metadata must be in .obs
        pseudobulk_key: str | List[str]
            Columns for pseudobulking counts. Usually this is by patient
        group_key: str | List[str]
            Metadata columns that define biological groups in which the analysis is performed
            Each combination of values in the group_key are independent pseudobulked datasets
            e.g. To compare within each primary cell type,
            group_key = 'primary_celltype'. To compare each secondary celltype within annotated regions,
            group_key = ['secondary_celltype', 'annot_region']
        comparisons: str | List[str]
            Column(s) to specify covariate(s) in the design matrix
        metadata: pd.DataFrame | None
            Dataframe containing the patient metadata. Must have columns for pseudobulk_key, group_key, and comparisons
        meta_merge_key: str
            Column to merge metadata and anndata.obs on
        is_grouped: bool
            Boolean to indicate that the anndata has been aggregated
        preselection: str | List[str]
            preselection criteria to filter patients using metadata.eval() (or adata.obs.eval())
        skip: defaultdict(list)
            Which groups to skip when doing comparisons. Keys must be present in group_key
            e.g. skip = {'primary_celltype': ['neural', 'muscle', 'Mast_cell']} skips comparisons using neural cells, muscle cells, and/or mast_cells
        adata_agg: AnnData | None (optional)
            Pre-aggregated anndata using
            ``sc.get.aggregate(adata, by = pseudobulk_key + group_key + comparisons, func = 'sum', layer = 'counts')``
            This is to transfer the aggregated count information between permutations
        design: str | None (optional)
            Design matrix formula. If None, use all comparisons and interactions between comparisons
        design_null: str | None (optional)
            Reduced design matrix formula, used to calculate reduced models for the log likelihood ratio tests
            If None, use a constant design matrix
        gene_set: str | None (optional)
            Path to gene set .txt file to create the gene_dict. If none, use the MSigDB_hallmark_2020
            pathway set compiled by Rick
        layer: str | None
            If not None, key for counts layer to aggregate data. If None, use anndata.X
        """
        
        self.adata = adata.copy()
        self.preselection = preselection if isinstance(preselection, list) else [preselection]
        self.pseudobulk_key = pseudobulk_key if isinstance(pseudobulk_key, list) else [pseudobulk_key]
        self.group_key = group_key if isinstance(group_key, list) else [group_key]
        self.comparisons = comparisons if isinstance(comparisons, list) else [comparisons]
        self.skip = skip
        self.metadata = metadata
        self.meta_merge_key = meta_merge_key
        self.is_grouped = is_grouped
        self.layer = layer
        
        if metadata is not None:
            cols = list(set(self.metadata.columns).intersection(set(self.pseudobulk_key + self.comparisons)))
            self.metadata = self.metadata[~self.metadata[cols].isna().any(axis = 1)]
            self.metadata = self.metadata[self.metadata[meta_merge_key].isin(self.adata.obs[meta_merge_key].unique())]
            if len(preselection) > 0:
                self.metadata = self.metadata[self.metadata.eval(' & '.join(f'({x})' for x in self.preselection))]
            idx = adata.obs.index.name
            # Merge the metadata with the observed, prioritizing the metadata if columns are shared
            merged_metadata = adata.obs.reset_index().merge(self.metadata, on = meta_merge_key, how = 'left', suffixes = ('_left', '')).set_index(idx)
            merged_metadata = merged_metadata.drop(columns = [c for c in merged_metadata.columns if '_left' in c])
            self.adata.obs = merged_metadata

        # Filter based on preselection and nan values for the given experimental design
        self.adata = self.adata[~self.adata.obs[self.pseudobulk_key + self.comparisons].isna().any(axis = 1)]
            
        if len(preselection) > 0:
            self.adata = self.adata[self.adata.obs.eval(' & '.join(f'({x})' for x in self.preselection))]
                
        self.is_grouped = is_grouped
        self.design = design if design is not None else "*".join(self.comparisons)
        self.design_null = design_null if design_null is not None else "1"
        
        self.dds = defaultdict() # DeseqDataSets for each group
        self.ds = defaultdict() # DeseqStats results_dfs for each group
        self.gsva_df = defaultdict() # GSVA dataframes
        self.gsva_models = defaultdict(dict) # GSVA OLS models
        self.gsva_results = defaultdict(dict) # GSVA OLS fit results
        self.gsva_null_models = defaultdict(dict) # GSVA OLS reduced models
        self.gsva_null_results = defaultdict(dict) # GSVA OLS reduced fit results
        
        self.results_df = None
        
        if gene_set is None:
            # Read in the gene set dictionary (ty Rick)
            self.gene_set = pd.read_json('/common/mebaner/xenium/h.all.v2025.1.Hs.json.txt')
        # TODO: Figure out how to accept multiple data formats later since we all use the same gene set
        else:
            self.gene_set = gene_set
        self.gene_dict = {}
        for c in self.gene_set.columns.values:
            self.gene_dict[c] = self.gene_set[c]['geneSymbols']  
            
    def run_deseq2(
        self,
        contrast = None,
        dds_kwargs = {}, 
        ds_kwargs = {},
        show_progress = True,
    ):
        """
        Run deseq2 analysis, store results in analyzer
        
        Parameters:
        ------------
        contrast : list  | np.ndarray | dict(dict)
            If a contrast is provided, the test will default to the Wald test. Otherwise defaults to
            the likelihood ratio test
            
            Documentation from PyDESeq2:
                Either a list of three strings or a numpy array.
                If a list of three strings, it must be in the following format:
                ``['variable_of_interest', 'tested_level', 'ref_level']``.
                Names must correspond to the metadata data passed to the DeseqDataSet.
                E.g., ``['condition', 'B', 'A']`` will measure the LFC of 'condition B' compared
                to 'condition A'.
                If a numpy array, it must be a contrast vector of the same length as the design
                matrix.   
                If it is a dictionary, it must have the keys 'test' and 'ref', whose values
                are dictionaries defining the reference and test classes respectively.
                E.g., ``{'test': dict(condition = 'B'), 'ref': dict(condition = 'A')}``
        dds_kwargs: dict
            kwargs for DeseqDataSet. Arguments shared with other kwargs will be overwritten
        ds_kwargs: dict
            kwargs for DeseqStats. Arguments shared with other kwargs will be overwritten
        layer: str | None
            Name of layer containing counts. If none, uses adata.X
        """
        
        # If a contrast is provided, pass it into the kwargs and use the wald test
        if contrast is not None:
            ds_kwargs['contrast'] = contrast
            ds_kwargs['test'] = 'Wald'
        # If no contrast is provided, use the likelihood ratio test if none is provided in the ds_kwargs
        else:
            if 'contrast' not in ds_kwargs:
                ds_kwargs['contrast'] = None
                ds_kwargs['test'] = 'LRT'
            elif ds_kwargs['contrast'] is None:
                ds_kwargs['test'] = 'LRT'
        if 'design_null' not in ds_kwargs:
            ds_kwargs['design_null'] = self.design_null
        
        t0 = time.time()
        
        # Aggregate if the anndata is not already aggregated
        if not self.is_grouped:
            if show_progress:
                print("Pseudobulking anndata ... ", end = "")
            adata_agg = sc.get.aggregate(self.adata, by = self.pseudobulk_key + self.group_key + self.comparisons, func = 'sum', layer = self.layer)
            self.adata = adata_agg
            self.is_grouped = True
            if show_progress:
                print(f"done: {(time.time() - t0)/60:.2f} min")
        
        for idx, row in (pbar := tqdm(self.adata.obs[self.group_key].drop_duplicates().iterrows(), total = len(self.adata.obs[self.group_key].drop_duplicates()), disable = not show_progress)):
            key = tuple([str(row[g]) for g in self.group_key])
            pbar.set_description(f"Running PyDeseq2 for {key}")
            
            # Check if group should be skipped
            skip_group = False
            for g in self.group_key:
                if row[g] in self.skip[g]:
                    skip_group = True
                    break
            if skip_group:
                continue
            # Get only cells in the given category:
            bdata = self.adata[self.adata.obs[self.group_key].eq(row).all(axis = 1)].copy()
            # Require >=3 patients with nonzero counts for each gene
            bdata = bdata[:, (bdata.layers['sum'] > 1).sum(axis = 0) >= 3].copy()
            # Make count dataframe for the deseq data set
            counts = pd.DataFrame(bdata.layers['sum'], columns = bdata.var_names)
            # Make the deseq dataset and run the statistical test
            dds = DeseqDataSet(
                counts = counts,
                metadata = bdata.obs,
                design = self.design,
                quiet = True,
                **dds_kwargs,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message = '.*dispersion trend curve fitting did not converge.*', category = UserWarning)
                dds.deseq2()
            self.dds[key] = dds
            stats = DeseqStats(
                dds, 
                quiet = True,
                **ds_kwargs)
            
            stats.summary()
            self.ds[key] = stats.results_df.copy()
    
    def run_gsva(self, gsva_kwargs = {}, show_progress = True):
        """
        Run the GSVA analysis. Must run Deseq2 analysis first to get normalized counts
        """
        for key in (pbar := tqdm(self.dds.keys(), disable = not show_progress)):
            pbar.set_description(f"Running GSVA for {key}")
            dds = self.dds[key]
            if 'vst_counts' not in dds.layers:
                dds.vst()
                
            counts = pd.DataFrame(dds.layers['vst_counts'], columns = dds.var_names, index = dds.obs.index)
            try:
                es = gp.gsva(
                    counts.T,
                    self.gene_dict,
                    **gsva_kwargs
                )
                
                res2d = es.res2d
                res2d['ES'] = res2d['ES'].astype(np.float32)
                pivot = es.res2d.pivot(index = "Name", columns = 'Term', values = 'ES')
                for c in self.pseudobulk_key:
                    pivot[c] = dds.obs[c]
                for c in self.comparisons:
                    pivot[c] = dds.obs[c]
                    
                self.gsva_df[key] = pivot
            except Exception as e:
                print(f"   GSVA fitting failed: {e}")
                traceback.print_exc()
                return None
                
    def fit_gsva(self, fit_kwargs = {}, show_progress = True):
        """
        After getting the gsva enrichment scores, fit OLS using the full and reduced models
        """
        
        for key in (pbar := tqdm(self.gsva_df.keys(), disable = not show_progress)):
            pbar.set_description(f"Fitting OLS to GSVA scores for {key}")
            df = self.gsva_df[key]

            cols_to_fit = [c for c in df.columns if c.startswith('HALLMARK')]
            
            for col in cols_to_fit:
                y, X = dmatrices(f'{col} ~ {self.design}', df, return_type = 'dataframe')
                X_null = FormulaicContrasts(df, self.design_null).design_matrix
                
                self.gsva_models[key][col] = OLS(y, X)
                self.gsva_results[key][col] = self.gsva_models[key][col].fit(**fit_kwargs)
                
                self.gsva_null_models[key][col] = OLS(y, X_null)
                self.gsva_null_results[key][col] = self.gsva_null_models[key][col].fit(**fit_kwargs)
    
    def run_stats_deseq(self):
        """
        Get the stat result dataframe from the Deseq2 analysis
        """
        out = pd.DataFrame()
        
        for key in self.ds:
            df = self.ds[key]
            for g, val in zip(self.group_key, key):
                df[g] = val
        
            out = pd.concat([out, df], axis = 0, join = 'outer')
        out = out.reset_index().rename(columns = {'index': 'feature'})
        if 'baseMean' in out.columns:
            out = out.drop(columns = ['baseMean'])
            out = out.rename(columns = {'log2FoldChange': 'effect', 'lfcSE': 'effectSE'})
        out['feature_type'] = 'gene'
        out['index'] = out.apply(lambda x: x['feature'] + '___' + '___'.join([f"{g}__{x[g]}" for g in self.group_key]), axis = 1)
        out = out.set_index('index')
        return out

    def run_stats_gsva(
        self,
        contrast = None
    ):
        """
        Calculate the stat results dataframe from the GSVA analysis
        
        Parameters:
        -----------
        contrast: np.ndarray | dict(dict) | None
            Contrast vector for the comparison If None, test will default to LRT.
            If dictionary, contrast must have keys 'test' and 'ref' with values given by
            dictionaries defining the test and reference classes
        """
        out = pd.DataFrame()
        
        # Loop over datasets
        for key in self.gsva_results:
            stats = []
            pvals = []
            pathways = []
            effects = []
            effectSE = []
            
            # If contrast is provided as a dictionary, use formulaic contrasts to get the vector
            if isinstance(contrast, dict):
                fc = FormulaicContrasts(self.gsva_df[key], f'~ {self.design}')
                contrast = (fc.cond(**contrast['test']) - fc.cond(**contrast['ref'])).values
                
            for pathway in self.gsva_results[key]:
                
                # If using likelihood ratios
                if contrast is None:
                    stat = max(0, 2 * (self.gsva_results[key][pathway].llf - self.gsva_null_results[key][pathway].llf))
                    p = chi2.sf(stat, df = self.gsva_models[key][pathway].exog.shape[1] - self.gsva_null_models[key][pathway].exog.shape[1])
                # Otherwise use the wald test (t-test is to get effect and errors)
                else:
                    res = self.gsva_results[key][pathway].wald_test(contrast, scalar = True)
                    stat = res.statistic
                    p = res.pvalue
                    res_t = self.gsva_results[key][pathway].t_test(contrast)
                    effects.append(res_t.effect.item())
                    effectSE.append(res_t.sd.item())
                stats.append(stat)
                pathways.append(pathway)
                pvals.append(p)
            padj = false_discovery_control(pvals)
            
            # Make dataframe for the given dataset
            df = pd.DataFrame({'stat': stats, 'feature': pathways, 'pvalue': pvals, 'padj': padj})
            if contrast is not None:
                df['effect'] = effects
                df['effectSE'] = effectSE
                
            # Add the group information to each dataframe
            for g, val in zip(self.group_key, key):
                df[g] = val
            # Concat to the output dataframe
            out = pd.concat([out, df], axis = 0, join = 'outer')
        out['feature_type'] = 'pathway'
        out['index'] = out.apply(lambda x: x['feature'] + '___' + '___'.join([f"{g}__{x[g]}" for g in self.group_key]), axis = 1)
        out = out.set_index('index')
        return out

    def run_stats(self, contrast = None):
        """
        Get the statistical results for the deseq2 and gsva analysis
        
        Parameters:
        -----------
        contrast: np.ndarray | dict(dict) | None
            Contrast vector for the comparison If None, test will default to LRT.
            If dictionary, contrast must have keys 'test' and 'ref' with values given by
            dictionaries defining the test and reference classes        
        """
        out_deseq = self.run_stats_deseq()
        out_gsva = self.run_stats_gsva(contrast = contrast)
        
        out_df = pd.concat([out_deseq, out_gsva], axis = 0, join = 'outer')
        self.results_df = out_df.copy()
        return out_df
    
    def run_analysis(
        self,
        n_jobs = None,
        contrast = None,
        dds_kwargs = {}, 
        ds_kwargs = {},
        gsva_kwargs = {},
        fit_kwargs = {},
        show_progress = True,
    ):
        """
        Run full analysis pipeline
        
        Parameters:
        -----------
        n_jobs: int | None
            Number of jobs for parallelization. If None, use all available cpus
        contrast: np.ndarray | dict(dict) | None
            Contrast vector for the comparison If None, test will default to LRT.
            If dictionary, contrast must have keys 'test' and 'ref' with values given by
            dictionaries defining the test and reference classes
        dds_kwargs: dict
            Keyword arguments for deseq2 data set construction
        ds_kwargs: dict
            Keyword arguments for deseq2 stats analysis
        gsva_kwargs: dict
            Keyword arguments for gsva analysis
        fit_kwargs: dict
            Keyword arguments for OLS fitting of GSVA results
        """
        
        # Sorting out kwargs
        dds_kwargs['n_cpus'] = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        ds_kwargs['n_cpus'] = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        gsva_kwargs['threads'] = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
        
        self.run_deseq2(show_progress=show_progress, contrast = contrast, dds_kwargs=dds_kwargs, ds_kwargs=ds_kwargs)
        self.run_gsva(gsva_kwargs=gsva_kwargs, show_progress=show_progress)
        self.fit_gsva(fit_kwargs=fit_kwargs, show_progress=show_progress)
        return self.run_stats(contrast = contrast)


    def _shuffle_metadata(self, random_state = None):
        """
        Shuffle patient metadata labels. Returns shuffled metadata dataframe
        
        Parameters:
        -----------
        random_state: int | None
            Random state for shuffling metadata for reproducibility
        """
        
        meta = self.adata.obs.copy() if self.metadata is None else self.metadata.copy()
        perm_cols = self.pseudobulk_key + self.comparisons
        shuffled_meta = meta[perm_cols].drop_duplicates()
        shuffled_meta['tmp_idx'] = shuffled_meta.apply(lambda x: '_'.join(str(x[c]) for c in perm_cols), axis = 1)
        
        rng = np.random.default_rng(random_state)
        shuffle_dict = dict(zip(shuffled_meta['tmp_idx'], rng.permutation(shuffled_meta['tmp_idx'])))
        shuffled_meta['tmp_idx'] = shuffled_meta['tmp_idx'].map(shuffle_dict)
        
        df = self.adata.obs.copy()
        df['tmp_idx'] = df.apply(lambda x: '_'.join(str(x[c]) for c in perm_cols), axis = 1)
        df = df.drop(columns = perm_cols)
        
        df = df.merge(shuffled_meta, on = 'tmp_idx', how = 'right').set_index(df.index)
        df = df.drop(columns = ['tmp_idx'])

        return df
    
    def _run_single_permutation(
        self,
        n_jobs = None,
        contrast = None,
        dds_kwargs = {}, 
        ds_kwargs = {},
        gsva_kwargs = {},
        fit_kwargs = {},
        random_state = None
    ):
        """
        Run analysis for single permutation
        
        Parameters:
        -----------
        n_jobs: int | None
            Number of jobs for parallelization. If None, use all available cpus
        contrast: np.ndarray | dict(dict) | None
            Contrast vector for the comparison. If None, test will default to LRT.
            If dictionary, contrast must have keys 'test' and 'ref' with values given by
            dictionaries defining the test and reference classes
        dds_kwargs: dict
            Keyword arguments for deseq2 data set construction
        ds_kwargs: dict
            Keyword arguments for deseq2 stats analysis
        gsva_kwargs: dict
            Keyword arguments for gsva analysis
        fit_kwargs: dict
            Keyword arguments for OLS fitting of GSVA results
        """
        adata = self.adata.copy()
        shuffled = self._shuffle_metadata(random_state = random_state)
        adata.obs = shuffled
        
        try:
            
            GA = GeneAnalyzer(
                adata,
                self.pseudobulk_key,
                self.group_key,
                self.comparisons,
                is_grouped = True,
                skip = self.skip,
                design = self.design,
                design_null = self.design_null,
                gene_set = self.gene_set
            )
            
            dds_kwargs['n_cpus'] = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
            ds_kwargs['n_cpus'] = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
            gsva_kwargs['threads'] = n_jobs if n_jobs is not None else multiprocessing.cpu_count()
            
            GA.run_deseq2(contrast = contrast, show_progress=False, dds_kwargs=dds_kwargs, ds_kwargs=ds_kwargs)
            GA.run_gsva(show_progress=False, gsva_kwargs=gsva_kwargs)
            GA.fit_gsva(show_progress=False, fit_kwargs=fit_kwargs)
            return GA.run_stats(contrast = contrast)
        
        except Exception as e:
            print(f"   Permutation fitting failed: {e}")
            traceback.print_exc()
            return None

    def _run_permutations(
        self,
        contrast = None,
        dds_kwargs = {}, 
        ds_kwargs = {},
        gsva_kwargs = {},
        fit_kwargs = {},
        n_permutations = 1000,
        n_jobs = None,
        random_state = None,
        show_progress = True
    ):
        
        if n_jobs is None:
            n_jobs = multiprocessing.cpu_count() - 1
            
        if n_jobs > 1:
            tasks = [delayed(self._run_single_permutation)(
                contrast = contrast,
                dds_kwargs = dds_kwargs,
                ds_kwargs = ds_kwargs,
                gsva_kwargs = gsva_kwargs,
                fit_kwargs = fit_kwargs,
                n_jobs = 1,
                random_state = i + random_state if random_state is not None else random_state
            ) for i in range(n_permutations)]
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message = '.*worker stopped while some jobs were given to the executor.*', category = UserWarning)
                warnings.filterwarnings("ignore", message = '.*dispersion trend curve fitting did not converge.*', category = UserWarning)

                with tqdm_joblib(tqdm(desc = "Calculating permutations", total = len(tasks), disable = not show_progress)) as pbar:
                    results = Parallel(n_jobs = n_jobs)(tasks)
                    
        else:
            results = [self._run_single_permutation(
                contrast = contrast,
                dds_kwargs = dds_kwargs,
                ds_kwargs = ds_kwargs,
                gsva_kwargs = gsva_kwargs,
                fit_kwargs = fit_kwargs,
                random_state = i + random_state if random_state is not None else random_state
            ) for i in tqdm(range(n_permutations), desc = "Calculating permutations")]

        out = pd.DataFrame()
        for n, perm_res in enumerate(results):
            if perm_res is None:
                continue
            df = perm_res.copy()
            df['perm_iter'] = n
            df = df[self.group_key + ['stat', 'perm_iter']]
            out = pd.concat([out, df], axis = 0, join = 'outer')
        return out
            
            
    def run_stats_with_permutations(
        self,
        contrast = None,
        dds_kwargs = {}, 
        ds_kwargs = {},
        gsva_kwargs = {},
        fit_kwargs = {},
        n_permutations = 1000,
        n_jobs = None,
        random_state = None,
        show_progress = True
    ):
        """
        Run all statistical tests using patient permutation testing. Returns the full stat analysis and permutation analysis dataframe
        
        Parameters:
        ------------
        skip: dict(list)
            Which groups to skip doing comparisons between.
            e.g. skip = {'primary_celltype': ['neural', 'muscle']} skips comparing between neural cells and muscle cells
        dds_kwargs: dict
            Keyword arguments for DeseqDataSet()
        ds_kwargs: dict
            Keyword arguments for DeseqStats()
        gsva_kwargs: dict
            Keyword arguments for gseapy.gsva()
        fit_kwargs: dict
            Keyword arguments for OLS fitting
        n_permutations: int
            Number of permutations to generate null distribution
        n_jobs: int | None
            Number of jobs for parallelization
        random_state: int
            Random state for permutation testing for reproducibility
        show_progress: bool
            Show tqdm progress bar for permutations
        """
        
        stat_res = self.run_stats(contrast = contrast) if self.results_df is None else self.results_df
        
        perm_res = self._run_permutations(
            contrast = contrast,
            dds_kwargs = dds_kwargs,
            ds_kwargs = ds_kwargs,
            gsva_kwargs = gsva_kwargs,
            fit_kwargs = fit_kwargs,
            n_permutations = n_permutations,
            n_jobs = n_jobs,
            random_state = random_state,
            show_progress = show_progress
        )
        
        stat_res = get_empirical_pvalues(stat_res, perm_res)

        out = {'df': stat_res, 'perm_df': perm_res}
        return out
        
    def plot_DEG(
        self,
        key,
        gene,
        ax = None,
        order = None,
        layer = 'normed_counts',
        logy = False,
        ylabel = None,
        title = "",
    ):
        """
        Function to plot differentially expressed genes across groups
        
        Parameters:
        ------------
        key: tuple
            Key that defines the specific combination of group_key values for the model
        gene: str
            Name of gene to plot
        ax: matplotlib.pyplot.axis | None
            Axis object to plot on (optional). If not provided, one will be created
        order: list | None
            Ordering for the cohorts on the x-axis. The classification is formatted as
            column1__value1___column2__value2___...etc for each col/val pair defining the comparison group
        layer: str | None
            Layer key to get count information from. Defaults to 'normed_counts' which are library size normalized
            values from deseq2. If None, use the raw count from dds.X
        """
        
        if ax is None:
            fig, ax = plt.subplots(constrained_layout = True)
            
        if layer == 'vst_counts' and layer not in self.dds[key].layers:
            self.dds[key].vst()
        
        dds = self.dds[key] 
        if hasattr(dds.var, 'refitted'):
            if self.dds[key].var['refitted'][gene]:
                dds = self.dds[key].counts_to_refit
        if gene not in dds.var_names:
            raise ValueError(f"Gene {gene} is not present in the deseq dataset")
        counts = dds[:,gene].layers[layer].flatten() if layer is not None or layer in dds[:, gene].layers else dds[:,gene].X.flatten()
        labels = dds.obs.apply(lambda x: "___".join([f"{c}__{x[c]}" for c in self.comparisons]), axis = 1)
        
        df = pd.DataFrame({gene: counts, 'classification': labels})
        
        sns.boxplot(
            x = 'classification',
            y = gene,
            order = order,
            hue = 'classification',
            hue_order = order,
            data = df,
            ax = ax,
        )
        
        sns.stripplot(
            x = 'classification',
            y = gene,
            order = order,
            hue = 'classification',
            hue_order = order,
            palette = 'dark:black',
            legend = False,
            s = 3,
            data = df,
            ax = ax,
        )
        
        if logy:
            ax.set_yscale('log')
        ax.set_title(gene)
        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            if layer == 'normed_counts':
                ax.set_ylabel("Normalized Counts")
            elif layer:
                ax.set_ylabel(layer)
            else:
                ax.set_ylabel("Counts")
        if title:
            ax.set_title(title)
        ax.grid()
        return ax
    
    def plot_gsva(
        self,
        key,
        pathway,
        order = None,
        ax = None,
        ylabel = None,
        title = ""
    ):
        """
        Function to plot GSVA enrichment scores for each cohort
        
        Parameters:
        ------------
        key: tuple
            Key that defines the specific combination of group_key values for the model
        pathway: str
            Name of pathway to plot
        order: list | None
            Ordering for the cohorts on the x-axis. The classification is formatted as
            column1__value1___column2__value2___...etc for each col/val pair defining the comparison group
        """
        
        if key not in self.gsva_df.keys():
            raise ValueError(key)
        elif pathway not in self.gsva_df[key].keys():
            raise ValueError(pathway)
        
        df = self.gsva_df[key].copy()
        
        if ax is None:
            fig, ax = plt.subplots(constrained_layout = True)
            
        df['classification'] = df.apply(lambda x: '___'.join(f"{c}__{x[c]}" for c in self.comparisons), axis = 1)
        
        sns.boxplot(
            x = 'classification',
            y = pathway,
            data = df,
            hue = 'classification',
            order = order,
            hue_order = order,
            ax = ax,
        )
        
        sns.stripplot(
            x = 'classification',
            y = pathway,
            data = df,
            hue = 'classification',
            palette = 'dark:black',
            s = 3,
            order = order,
            hue_order = order,
            ax = ax,
            legend = False
        )
        
        ax.set_ylabel("GSVA Enrichment Score" if ylabel is None else ylabel)
        ax.set_title(pathway if title is None else title)
        ax.grid()
        return ax
    
    def plot_feature(self, feature, **kwargs):
        """
        Function to plot any feature
        
        Parameters:
        -----------
        feature: str
            Name of the feature given from the stat results dataframe (self.results_df)
        **kwargs:
            Keyword arguments for the plotting functions
        """

        feat = feature.split('___')[0]
        key = tuple(f.split('__')[-1] for f in feature.split('___')[1:])
        feature_type: str
        if self.results_df is not None:
            try:
                feature_type = self.results_df.loc[feature, 'feature_type']
            except:
                print(f"Feature {feature} has not been added")
                return
        else:
            feature_type = 'pathway' if 'HALLMARK' in feature else 'gene'
            
        if feature_type == 'pathway':
            return self.plot_gsva(key, feat, **kwargs)
        elif feature_type == 'gene':
            return self.plot_DEG(key, feat, **kwargs)
        else:
            raise ValueError(f"Unrecognized feature type from feature name: {feature_type}")
        
    def get_feature_dataframe(self, feature):
        feat = feature.split('___')[0]
        key = tuple(f.split('__')[-1] for f in feature.split('___')[1:])
        feature_type: str
        if self.results_df is not None:
            try:
                feature_type = self.results_df.loc[feature, 'feature_type']
            except:
                print(f"Feature {feature} has not been added")
                return
        else:
            feature_type = 'pathway' if 'HALLMARK' in feature else 'gene'
        if feature_type == 'gene':
            dds = self.dds[key]
            counts = dds[:, feat].layers['normed_counts'].flatten()
            labels = dds.obs.apply(lambda x: "___".join([f"{c}__{x[c]}" for c in self.comparisons]), axis = 1)
            
            df = pd.DataFrame({feat: counts, 'classification': labels})

            return df
        elif feature_type == 'pathway':
            if key not in self.gsva_df.keys():
                raise ValueError(key)
            elif feat not in self.gsva_df[key].keys():
                raise ValueError(feat)
        
            df = self.gsva_df[key].copy()
            df['classification'] = df.apply(lambda x: '___'.join(f"{c}__{x[c]}" for c in self.comparisons), axis = 1)

            return df[[feat, 'classification']]