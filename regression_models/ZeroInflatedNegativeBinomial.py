from statsmodels_utils import *

class ZeroInflatedNegativeBinomialCustomResults(ZeroInflatedNegativeBinomialResults):
    """
    Wrapper class for zero inflated negative binomial model results
    """
    def __init__(
        self, 
        CountResults,
        **kwargs):
        
        super(ZeroInflatedNegativeBinomialCustomResults, self).__init__(CountResults, **kwargs)

    def get_pdf(self, x, contrast):
        beta = self.params[1:-1]
        phi = self.params[-1]
        mu = self.model.family.link.inverse((beta * contrast).sum())
        zinf = sm.genmod.families.links.Logit().inverse(self.params[0])
        pdf = (1 - zinf) * stats.gamma.pdf(x, a = 1 / phi, scale = mu * phi)
        zero_bin = zinf / (x[1] - x[0])
        pdf[0] = pdf[0] + zero_bin # Add the zero inflation to the first bin (even if it's not zero)
        return pdf

    
class ZeroInflatedNegativeBinomialCustom(ZeroInflatedNegativeBinomialP):
    def __init__(self, 
                 endog,
                 exog, 
                 link = sm.genmod.families.links.Log(),
                 **kwargs):
        super(ZeroInflatedNegativeBinomialCustom, self).__init__(endog, exog, **kwargs)
        
        # These are not needed for fitting but used to generalize functions in the wrapper class
        self.family = sm.genmod.families.NegativeBinomial(alpha = 1.0, link = link)
        self.link = link
        
    def fit(self, *args, **kwargs):
        # Count models print info by default unless you tell them not to
        if 'disp' not in kwargs:
            kwargs['disp'] = 0
        res = super().fit(*args, **kwargs)
        res._results.__class__ = ZeroInflatedNegativeBinomialCustomResults
        return res
