from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        self.pi_ = np.zeros(self.classes_.size)
        for i, k in enumerate(self.classes_):
            self.mu_[i] = X[y == k].mean(axis=0)
            self.pi_[i] = sum(y == k) / X.shape[0]
            sigma_k = X[y == k] - self.mu_[i]
            self.cov_ += sigma_k.T @ sigma_k
        self.cov_ /= X.shape[0] - self.classes_.size
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihood = np.zeros((X.shape[0], self.classes_.size))
        # likelihood[i,k] = log-likelihood of X[i]'s label to be the k-th class
        for i in range(X.shape[0]):
            for k in self.classes_:
                a_k = self._cov_inv @ self.mu_[k]
                b_k = np.log(self.pi_[k]) - 0.5 * (self.mu_[k] @
                                                   self._cov_inv @ self.mu_[k])
                likelihood[i, k] = a_k @ X[i] + b_k
        # # if explicit likelihood is requested instead of log-likelihood,
        # # following lines can be used:
        # # likelihood[i,k] = likelihood of X[i]'s label to be the k-th class.
        # for i in range(X.shape[0]):
        #     for k in self.classes_.size:
        #         x_i_m_mu_k = X[i] - self.mu_[k]
        #         denominator = (2 * np.pi) ** X.shape[1] * det(self.cov_)
        #         likelihood[i, k] = self.pi_[k] / denominator * np.exp(
        #             -0.5 * x_i_m_mu_k.T @ inv(self.cov_) @ x_i_m_mu_k)
        return likelihood

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
