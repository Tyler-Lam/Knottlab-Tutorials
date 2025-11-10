from statsmodels_utils import *

class BetaGLMResults(BetaResults):
    """
    Wrapper class for Beta model results
    """
    def __init__(self, model, mlefit):
        super(BetaGLMResults, self).__init__(model, mlefit)
        
    # Use the model's residual function
    def resid(self, which = 'rate'):
        if which == 'rate':
            return super().resid
        elif which == 'pearson':
            return self.resid_pearson
    
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
    
    # Get expected output given an exog
    def predict(self, params, exog = None, which = 'counts'):

        if exog is None:
            exog = self.exog

        beta = params[:-1]
        phi = self.link_precision.inverse(params[-1])
        eta = exog @ beta
        mu = self.link.inverse(eta)

        if which == 'rate': # Return expected proportions
            return mu
        elif which == 'linear': # Return linear predictors (not inv-link transformed)
            return eta
        elif which == 'var':
            return super().predict(params, exog = exog, which = 'var')
        else:
            raise ValueError("predict parameter 'which' must be 'rate', 'var', or 'linear'")
        
    def resid(self, params, exog = None, endog = None, which = 'rate'):

        if exog is None:
            exog = self.exog
        if endog is None:
            endog = self.endog

        if which == 'rate':
            y_pred = self.predict(params = params, exog = exog, which = which)
            return endog - y_pred
        elif which == 'pearson':
            y_pred = self.predict(params = params, exog = exog, which = 'mean')
            var_pred = self.predict(params = params, exog = exog, which = 'var')
            return (endog - y_pred) / np.sqrt(var_pred)
        elif which == 'linear':
            y_pred = self.predict(params = params, exog = exog, which = which)
            return self.link(endog) - y_pred
        else:
            raise ValueError("resid parameter 'which' must be 'rate', 'pearson', or 'linear'")