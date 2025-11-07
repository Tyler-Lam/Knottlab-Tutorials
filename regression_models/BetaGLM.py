from statsmodels_utils import *

class BetaGLMResults(BetaResults):
    """
    Wrapper class for Beta model results
    """
    def __init__(self, model, mlefit):
        super(BetaGLMResults, self).__init__(model, mlefit)
        
    def get_pdf(self, x, contrast):
        params = self.params
        beta = params[:-1]
        phi = self.model.link_precision.inverse(params[-1])
        mu = self.model.link.inverse((beta * contrast).sum())
        a = mu * phi
        b = (1 - mu) * phi
        return stats.beta.pdf(x, a, b, loc = 0, scale = 1)

class BetaGLM(BetaModel):
    def __init__(self, endog, exog, **kwargs):
        super(BetaGLM, self).__init__(endog, exog, **kwargs)
        
    def fit(self, *args, **kwargs):
        res = super().fit(*args, **kwargs)
        res._results.__class__ = BetaGLMResults
        return res