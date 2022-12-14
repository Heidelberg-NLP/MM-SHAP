from .deep_pytorch import PyTorchDeep
from .deep_tf import TFDeep
from .._explainer import Explainer


class Deep(Explainer):
    """ Meant to approximate SHAP values for deep learning models.

    This is an enhanced version of the DeepLIFT algorithm (Deep SHAP) where, similar to Kernel SHAP, we
    approximate the conditional expectations of SHAP values using a selection of background samples.
    Lundberg and Lee, NIPS 2017 showed that the per node attribution rules in DeepLIFT (Shrikumar,
    Greenside, and Kundaje, arXiv 2017) can be chosen to approximate Shapley values. By integrating
    over many backgound samples Deep estimates approximate SHAP values such that they sum
    up to the difference between the expected model output on the passed background samples and the
    current model output (f(x) - E[f(x)]).

    Examples
    --------
    See :ref:`Deep Explainer Examples <deep_explainer_examples>`
    """

    def __init__(self, model, data, session=None, learning_phase_flags=None):
        """ An explainer object for a differentiable model using a given background dataset.

        Note that the complexity of the method scales linearly with the number of background data
        samples. Passing the entire training dataset as `data` will give very accurate expected
        values, but be unreasonably expensive. The variance of the expectation estimates scale by
        roughly 1/sqrt(N) for N background data samples. So 100 samples will give a good estimate,
        and 1000 samples a very good estimate of the expected values.

        Parameters
        ----------
        model : if framework == 'tensorflow', (input : [tf.Tensor], output : tf.Tensor)
             A pair of TensorFlow tensors (or a list and a tensor) that specifies the input and
            output of the model to be explained. Note that SHAP values are specific to a single
            output value, so the output tf.Tensor should be a single dimensional output (,1).

            if framework == 'pytorch', an nn.Module object (model), or a tuple (model, layer),
                where both are nn.Module objects
            The model is an nn.Module object which takes as input a tensor (or list of tensors) of
            shape data, and returns a single dimensional output.
            If the input is a tuple, the returned shap values will be for the input of the
            layer argument. layer must be a layer in the model, i.e. model.conv2

        data :
            if framework == 'tensorflow': [numpy.array] or [pandas.DataFrame]
            if framework == 'pytorch': [torch.tensor]
            The background dataset to use for integrating out features. Deep integrates
            over these samples. The data passed here must match the input tensors given in the
            first argument. Note that since these samples are integrated over for each sample you
            should only something like 100 or 1000 random background samples, not the whole training
            dataset.

        if framework == 'tensorflow':

        session : None or tensorflow.Session
            The TensorFlow session that has the model we are explaining. If None is passed then
            we do our best to find the right session, first looking for a keras session, then
            falling back to the default TensorFlow session.

        learning_phase_flags : None or list of tensors
            If you have your own custom learning phase flags pass them here. When explaining a prediction
            we need to ensure we are not in training mode, since this changes the behavior of ops like
            batch norm or dropout. If None is passed then we look for tensors in the graph that look like
            learning phase flags (this works for Keras models). Note that we assume all the flags should
            have a value of False during predictions (and hence explanations).
        """
        # first, we need to find the framework
        if type(model) is tuple:
            a, b = model
            try:
                a.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'
        else:
            try:
                model.named_parameters()
                framework = 'pytorch'
            except:
                framework = 'tensorflow'

        if framework == 'tensorflow':
            self.explainer = TFDeep(model, data, session, learning_phase_flags)
        elif framework == 'pytorch':
            self.explainer = PyTorchDeep(model, data)

        self.expected_value = self.explainer.expected_value

    def shap_values(self, X, ranked_outputs=None, output_rank_order='max', check_additivity=True):
        """ Return approximate SHAP values for the model applied to the data given by X.

        Parameters
        ----------
        X : list,
            if framework == 'tensorflow': numpy.array, or pandas.DataFrame
            if framework == 'pytorch': torch.tensor
            A tensor (or list of tensors) of samples (where X.shape[0] == # samples) on which to
            explain the model's output.

        ranked_outputs : None or int
            If ranked_outputs is None then we explain all the outputs in a multi-output model. If
            ranked_outputs is a positive integer then we only explain that many of the top model
            outputs (where "top" is determined by output_rank_order). Note that this causes a pair
            of values to be returned (shap_values, indexes), where shap_values is a list of numpy
            arrays for each of the output ranks, and indexes is a matrix that indicates for each sample
            which output indexes were choses as "top".

        output_rank_order : "max", "min", or "max_abs"
            How to order the model outputs when using ranked_outputs, either by maximum, minimum, or
            maximum absolute value.

        Returns
        -------
        array or list
            For a models with a single output this returns a tensor of SHAP values with the same shape
            as X. For a model with multiple outputs this returns a list of SHAP value tensors, each of
            which are the same shape as X. If ranked_outputs is None then this list of tensors matches
            the number of model outputs. If ranked_outputs is a positive integer a pair is returned
            (shap_values, indexes), where shap_values is a list of tensors with a length of
            ranked_outputs, and indexes is a matrix that indicates for each sample which output indexes
            were chosen as "top".
        """
        return self.explainer.shap_values(X, ranked_outputs, output_rank_order, check_additivity=check_additivity)
