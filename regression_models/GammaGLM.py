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

    def get_pdf(self, x, contrast):
        beta = self.params
        phi = self.model.scale
        mu = self.model.family.link.inverse((beta * contrast).sum())
        return stats.gamma.pdf(x, a = 1 / phi, scale = mu * phi)

    
class GammaGLM(sm.GLM):
    def __init__(self, 
                 endog,
                 exog, 
                 link = sm.genmod.families.links.Log(),
                 **kwargs):
        super(GammaGLM, self).__init__(endog, exog, family = sm.genmod.families.Gamma(link = link), **kwargs)
        self.link = link
    def fit(self, *args, **kwargs):
        res = super().fit(*args, **kwargs)
        res._results.__class__ = GammaGLMResults
        return res