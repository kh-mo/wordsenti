from abc import ABCMeta, abstractmethod

import warnings
import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors.unsupervised import NearestNeighbors
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.exceptions import ConvergenceWarning

class BaseLabelPropagation(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):
    """Base class for label propagation module.

    Parameters
    ----------
    kernel : {'knn', 'rbf', callable}
        String identifier for kernel function to use or the kernel function
        itself. Only 'rbf' and 'knn' strings are valid inputs. The function
        passed should take two inputs, each of shape [n_samples, n_features],
        and return a [n_samples, n_samples] shaped weight matrix

    gamma : float
        Parameter for rbf kernel

    n_neighbors : integer > 0
        Parameter for knn kernel

    alpha : float
        Clamping factor

    max_iter : integer
        Change maximum number of iterations allowed

    tol : float
        Convergence tolerance: threshold to consider the system at steady
        state

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    """

    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7, alpha=None, max_iter=30, tol=1e-3, n_jobs=1):

        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.n_jobs = n_jobs

    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(self.n_neighbors,
                                               n_jobs=self.n_jobs).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(self.nn_fit._fit_X,
                                                    self.n_neighbors,
                                                    mode='connectivity')
            else:
                return self.nn_fit.kneighbors(y, return_distance=False)
        elif callable(self.kernel):
            if y is None:
                return self.kernel(X, X)
            else:
                return self.kernel(X, y)
        else:
            raise ValueError("%s is not a valid kernel. Only rbf and knn"
                             " or an explicit function "
                             " are supported at this time." % self.kernel)

    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError("Graph construction must be implemented"
                                  " to fit a label propagation model.")

    def predict(self, X):
        """Performs inductive inference across the model.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input data
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()

    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Compute the probability estimates for each single sample in X
        and each possible outcome seen during training (categorical
        distribution).

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        probabilities : array, shape = [n_samples, n_classes]
            Normalized probability distributions across
            class labels
        """
        check_is_fitted(self, 'X_')

        X_2d = check_array(X, accept_sparse=['csc', 'csr', 'coo', 'dok',
                                             'bsr', 'lil', 'dia'])
        weight_matrices = self._get_kernel(self.X_, X_2d)
        if self.kernel == 'knn':
            probabilities = []
            for weight_matrix in weight_matrices:
                ine = np.sum(self.label_distributions_[weight_matrix], axis=0)
                probabilities.append(ine)
            probabilities = np.array(probabilities)
        else:
            weight_matrices = weight_matrices.T
            probabilities = np.dot(weight_matrices, self.label_distributions_)
        normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
        probabilities /= normalizer
        return probabilities

    def fit(self, X, y):
        """Fit a semi-supervised label propagation model based

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y)
        self.X_ = X
        check_classification_targets(y)

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph() # kh-mo : distance matrix (46473 * 46473)

        # label construction
        # construct a categorical distribution for classification only
        classes = np.unique(y) # kh-mo : classes [-1, 0, 1]
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)

        alpha = self.alpha
        if self._variant == 'spreading' and (alpha is None or alpha <= 0.0 or alpha >= 1.0):
            raise ValueError('alpha=%s is invalid: it must be inside '
                             'the open interval (0, 1)' % alpha)
        y = np.asarray(y)
        unlabeled = y == 0

        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1

        y_static = np.copy(self.label_distributions_)
        if self._variant == 'propagation':
            # LabelPropagation
            y_static[unlabeled] = 0
        else:
            # LabelSpreading
            y_static *= 1 - alpha

        l_previous = np.zeros((self.X_.shape[0], n_classes))

        unlabeled = unlabeled[:, np.newaxis]
        if sparse.isspmatrix(graph_matrix):
            graph_matrix = graph_matrix.tocsr()

        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_ ## store temporary label
            self.label_distributions_ = safe_sparse_dot(graph_matrix, self.label_distributions_) ## label dot product

            if self._variant == 'propagation':
                normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
                self.label_distributions_ /= normalizer
                self.label_distributions_ = np.where(unlabeled,
                                                     self.label_distributions_,
                                                     y_static)
            else:
                # clamp
                self.label_distributions_ = np.multiply(
                    alpha, self.label_distributions_) + y_static
        else:
            warnings.warn(
                'max_iter=%d was reached without convergence.' % self.max_iter,
                category=ConvergenceWarning
            )
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        self.label_distributions_ /= normalizer

        # set the transduction item
        transduction = self.classes_[np.argmax(self.label_distributions_,
                                               axis=1)]
        self.transduction_ = transduction.ravel()
        return self

class LabelPropagation(BaseLabelPropagation):
    _variant = 'propagation'

    def __init__(self, kernel='rbf', gamma=20, n_neighbors=7, alpha=1, max_iter=1000, tol=1e-3, n_jobs=1):
        super(LabelPropagation, self).__init__(kernel=kernel, gamma=gamma, n_neighbors=n_neighbors, alpha=alpha,
                                               max_iter=max_iter, tol=tol, n_jobs=n_jobs)
        self.affinity_matrix = None

    def _build_graph(self):
        return self.affinity_matrix
    # def _build_graph(self):
    #     """Matrix representing a fully connected graph between each sample
    #
    #     This basic implementation creates a non-stochastic affinity matrix, so
    #     class distributions will exceed 1 (normalization may be desired).
    #     """
    #     if self.kernel == 'knn':
    #         self.nn_fit = None
    #     affinity_matrix = self._get_kernel(self.X_)
    #     normalizer = affinity_matrix.sum(axis=0)
    #     if sparse.isspmatrix(affinity_matrix):
    #         affinity_matrix.data /= np.diag(np.array(normalizer))
    #     else:
    #         affinity_matrix /= normalizer[:, np.newaxis]
    #     return affinity_matrix

    def fit(self, X, y):
        if self.alpha is not None:
            warnings.warn("alpha is deprecated since 0.19 and will be removed in 0.21.", DeprecationWarning)
            self.alpha = None
        return super(LabelPropagation, self).fit(X, y)