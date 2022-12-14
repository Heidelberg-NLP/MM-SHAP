from .._explainer import Explainer
import numpy as np

class Coefficent(Explainer):
    """ Simply returns the model coefficents as the feature attributions.

    This is only for benchmark comparisons and does not approximate SHAP values in a
    meaningful way.
    """
    def __init__(self, model):
        assert hasattr(model, "coef_"), "The passed model does not have a coef_ attribute!"
        self.model = model

    def attributions(self, X):
        return np.tile(self.model.coef_, (X.shape[0], 1))
