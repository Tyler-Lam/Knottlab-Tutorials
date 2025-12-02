from statsmodels_utils import *

class NegativeBinomialCustomResults(NegativeBinomialResults):
    """
    Wrapper class for negative binomial model results so we can get the pdf from a contrast
    """
    def __init__(
        self, 
        CountResults,
        **kwargs):
        
        super(NegativeBinomialCustomResults, self).__init__(CountResults, **kwargs)

    def get_pdf(self, x, contrast):
        beta = self.params[:-1]
        phi = self.params[-1]
        mu = self.model.family.link.inverse((beta * contrast).sum())
        return stats.gamma.pdf(x, a = 1 / phi, scale = mu * phi)

    
class NegativeBinomialCustom(NegativeBinomial):
    def __init__(self, 
                 endog,
                 exog, 
                 link = sm.genmod.families.links.Log(),
                 **kwargs):
        super(NegativeBinomialCustom, self).__init__(endog, exog, **kwargs)
        
        # These are not needed for fitting but used to generalize functions in the wrapper class
        self.family = sm.genmod.families.NegativeBinomial(alpha = 1.0, link = link)
        self.link = link
        
    def fit(self, *args, **kwargs):
        # Count models print info by default unless you tell them not to
        if 'disp' not in kwargs:
            kwargs['disp'] = 0
        res = super().fit(*args, **kwargs)
        res._results.__class__ = NegativeBinomialCustomResults
        return res
