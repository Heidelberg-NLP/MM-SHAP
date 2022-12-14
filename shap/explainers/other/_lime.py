from .._explainer import Explainer
import numpy as np
try:
    import lime
    import lime.lime_tabular    
except ImportError:
    pass

class LimeTabular(Explainer):
    """ Simply wrap of lime.lime_tabular.LimeTabularExplainer into the common shap interface.

    Parameters
    ----------
    model : function or iml.Model
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array
        The background dataset. 

    mode : "classification" or "regression"
        Control the mode of LIME tabular.
    """

    def __init__(self, model, data, mode="classification"):
        self.model = model
        assert mode in ["classification", "regression"]
        self.mode = mode

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(data, mode=mode)
        
        out = self.model(data[0:1])
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
            if mode == "classification":
                def pred(X): # assume that 1d outputs are probabilities
                    preds = self.model(X).reshape(-1, 1)
                    p0 = 1 - preds
                    return np.hstack((p0, preds))
                self.model = pred 
        else:
            self.out_dim = self.model(data[0:1]).shape[1]
            self.flat_out = False

    def attributions(self, X, nsamples=5000, num_features=None):
        num_features = X.shape[1] if num_features is None else num_features
        
        if str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
            
        out = [np.zeros(X.shape) for j in range(self.out_dim)]
        for i in range(X.shape[0]):
            exp = self.explainer.explain_instance(X[i], self.model, labels=range(self.out_dim), num_features=num_features)
            for j in range(self.out_dim):
                for k,v in exp.local_exp[j]:
                    out[j][i,k] = v
        
        # because it output two results even for only one model output, and they are negated from what we expect
        if self.mode == "regression":
            for i in range(len(out)):
                out[i] = -out[i]
        
        return out[0] if self.flat_out else out
