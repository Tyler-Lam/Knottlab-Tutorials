from statsmodels_utils import *

class GammaGLMResults(GLMResults):
    """
    Wrapper class for Gamma model results
    """
    def __init__(
        self, 
        model,
        params, 
        normalized_cov_params,
        scale, 
        **kwargs):
        
        super(GammaGLMResults, self).__init__(model, params, normalized_cov_params, scale, **kwargs)
        
    # Use the model's residual function
    def resid(self, which = 'rate'):
        if which == 'rate':
            return self.resid_response
        elif which == 'pearson':
            return self.resid_pearson
    
    def get_pdf(self, x, contrast):
        beta = self.params
        phi = self.model.scale
        mu = self.model.family.link.inverse((beta * contrast).sum())
        return stats.gamma.pdf(x, a = 1 / phi, scale = mu * phi)

    
class GammaGLM(sm.GLM):
    def __init__(self, 
                 endog,
                 exog, 
                 **kwargs):
        super(GammaGLM, self).__init__(endog, exog, family = sm.genmod.families.Gamma(link = sm.genmod.families.links.Log()), **kwargs)
        
    def fit(self, *args, **kwargs):
        res = super().fit(*args, **kwargs)
        res._results.__class__ = GammaGLMResults
        return res
    
    # Get expected output given an exog
    def predict(self, params, exog = None, which = 'mean', **kwargs):

        if exog is None:
            exog = self.exog
            
        if which == 'rate' or which == 'mean': # Return expected proportions
            return super().predict(params, exog = exog, which = 'mean')
        elif which == 'linear': # Return linear predictors (not inv-link transformed)
            return self.family.link(super().predict(params, exog = exog, which = 'mean'))
        elif which == 'var' or which == 'var_unscaled':
            return super().predict(params, exog = exog, which = 'var_unscaled')
        else:
            return super().predict(params, exog = exog, which = which, **kwargs)
        
    def resid(self, params, exog = None, endog = None, which = 'mean', **kwargs):

        if exog is None:
            exog = self.exog
        if endog is None:
            endog = self.endog

        if which == 'rate' or which == 'mean':
            y_pred = super().predict(params = params, exog = exog, which = 'mean', **kwargs)
            return endog - y_pred
        elif which == 'pearson':
            y_pred = self.predict(params = params, exog = exog, which = 'mean', **kwargs)
            var_pred = self.predict(params = params, exog = exog, which = 'var', **kwargs)
            return (endog[:,0] - y_pred) / np.sqrt(var_pred)
        elif which == 'linear':
            y_pred = self.predict(params = params, exog = exog, which = 'mean', **kwargs)
            return self.family.link(endog) - y_pred
        else:
            raise ValueError("resid parameter 'which' must be 'mean', 'rate', 'pearson', or 'linear'")