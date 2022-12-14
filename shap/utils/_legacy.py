import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.sparse import issparse


def kmeans(X, k, round_values=True):
    """ Summarize a dataset with k mean samples weighted by the number of data points they
    each represent.

    Parameters
    ----------
    X : numpy.array or pandas.DataFrame or any scipy.sparse matrix
        Matrix of data samples to summarize (# samples x # features)

    k : int
        Number of means to use for approximation.

    round_values : bool
        For all i, round the ith dimension of each mean sample to match the nearest value
        from X[:,i]. This ensures discrete features always get a valid value.

    Returns
    -------
    DenseData object.
    """

    group_names = [str(i) for i in range(X.shape[1])]
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        group_names = X.columns
        X = X.values

    # in case there are any missing values in data impute them
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = X[:,j].toarray().flatten() if issparse(X) else X[:, j] # sparse support courtesy of @PrimozGodec
                ind = np.argmin(np.abs(xj - kmeans.cluster_centers_[i,j]))
                kmeans.cluster_centers_[i,j] = X[ind,j]
    return DenseData(kmeans.cluster_centers_, group_names, None, 1.0*np.bincount(kmeans.labels_))


class Instance:
    def __init__(self, x, group_display_values):
        self.x = x
        self.group_display_values = group_display_values


def convert_to_instance(val):
    if isinstance(val, Instance):
        return val
    else:
        return Instance(val, None)


class InstanceWithIndex(Instance):
    def __init__(self, x, column_name, index_value, index_name, group_display_values):
        Instance.__init__(self, x, group_display_values)
        self.index_value = index_value
        self.index_name = index_name
        self.column_name = column_name

    def convert_to_df(self):
        index = pd.DataFrame(self.index_value, columns=[self.index_name])
        data = pd.DataFrame(self.x, columns=self.column_name)
        df = pd.concat([index, data], axis=1)
        df = df.set_index(self.index_name)
        return df


def convert_to_instance_with_index(val, column_name, index_value, index_name):
    return InstanceWithIndex(val, column_name, index_value, index_name, None)


def match_instance_to_data(instance, data):
    assert isinstance(instance, Instance), "instance must be of type Instance!"

    if isinstance(data, DenseData):
        if instance.group_display_values is None:
            instance.group_display_values = [instance.x[0, group[0]] if len(group) == 1 else "" for group in data.groups]
        assert len(instance.group_display_values) == len(data.groups)
        instance.groups = data.groups


class Model:
    def __init__(self, f, out_names):
        self.f = f
        self.out_names = out_names


def convert_to_model(val):
    if isinstance(val, Model):
        return val
    else:
        return Model(val, None)


def match_model_to_data(model, data):
    assert isinstance(model, Model), "model must be of type Model!"
    
    try:
        if isinstance(data, DenseDataWithIndex):
            out_val = model.f(data.convert_to_df())
        else:
            out_val = model.f(data.data)
    except:
        print("Provided model function fails when applied to the provided data set.")
        raise

    if model.out_names is None:
        if len(out_val.shape) == 1:
            model.out_names = ["output value"]
        else:
            model.out_names = ["output value "+str(i) for i in range(out_val.shape[0])]
    
    return out_val



class Data:
    def __init__(self):
        pass


class SparseData(Data):
    def __init__(self, data, *args):
        num_samples = data.shape[0]
        self.weights = np.ones(num_samples)
        self.weights /= np.sum(self.weights)
        self.transposed = False
        self.groups = None
        self.group_names = None
        self.groups_size = data.shape[1]
        self.data = data


class DenseData(Data):
    def __init__(self, data, group_names, *args):
        self.groups = args[0] if len(args) > 0 and args[0] is not None else [np.array([i]) for i in range(len(group_names))]

        l = sum(len(g) for g in self.groups)
        num_samples = data.shape[0]
        t = False
        if l != data.shape[1]:
            t = True
            num_samples = data.shape[1]

        valid = (not t and l == data.shape[1]) or (t and l == data.shape[0])
        assert valid, "# of names must match data matrix!"

        self.weights = args[1] if len(args) > 1 else np.ones(num_samples)
        self.weights /= np.sum(self.weights)
        wl = len(self.weights)
        valid = (not t and wl == data.shape[0]) or (t and wl == data.shape[1])
        assert valid, "# weights must match data matrix!"

        self.transposed = t
        self.group_names = group_names
        self.data = data
        self.groups_size = len(self.groups)


class DenseDataWithIndex(DenseData):
    def __init__(self, data, group_names, index, index_name, *args):
        DenseData.__init__(self, data, group_names, *args)
        self.index_value = index
        self.index_name = index_name

    def convert_to_df(self):
        data = pd.DataFrame(self.data, columns=self.group_names)
        index = pd.DataFrame(self.index_value, columns=[self.index_name])
        df = pd.concat([index, data], axis=1)
        df = df.set_index(self.index_name)
        return df


def convert_to_data(val, keep_index=False):
    if isinstance(val, Data):
        return val
    elif type(val) == np.ndarray:
        return DenseData(val, [str(i) for i in range(val.shape[1])])
    elif str(type(val)).endswith("'pandas.core.series.Series'>"):
        return DenseData(val.values.reshape((1,len(val))), list(val.index))
    elif str(type(val)).endswith("'pandas.core.frame.DataFrame'>"):
        if keep_index:
            return DenseDataWithIndex(val.values, list(val.columns), val.index.values, val.index.name)
        else:
            return DenseData(val.values, list(val.columns))
    elif sp.sparse.issparse(val):
        if not sp.sparse.isspmatrix_csr(val):
            val = val.tocsr()
        return SparseData(val)
    else:
        assert False, "Unknown type passed as data object: "+str(type(val))

class Link:
    def __init__(self):
        pass


class IdentityLink(Link):
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x






class LogitLink(Link):
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return np.log(x/(1-x))

    @staticmethod
    def finv(x):
        return 1/(1+np.exp(-x))


def convert_to_link(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLink()
    elif val == "logit":
        return LogitLink()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"