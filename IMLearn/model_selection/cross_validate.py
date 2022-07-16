from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    divided_x = np.array_split(X, cv)
    divided_y = np.array_split(y, cv)
    train_scores = np.zeros(cv)
    validation_scores = np.zeros(cv)
    for i in range(cv):
        train_X = np.concatenate(np.delete(divided_x, i, 0))
        train_y = np.concatenate(np.delete(divided_y, i, 0))
        estimator.fit(train_X, train_y)
        train_scores[i] = scoring(train_y, estimator.predict(train_X))
        validation_scores[i] = scoring(divided_y[i],
                                       estimator.predict(divided_x[i]))
    return np.average(train_scores), np.average(validation_scores)
