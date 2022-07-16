from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

POLYNOMIAL_MAX_DEGREE = 10


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    f_X = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    epsilon = np.random.normal(0, noise, n_samples)
    train_X, train_y, \
    test_X, test_y = split_train_test(pd.DataFrame(X),
                                      pd.Series(f_X + epsilon), 2 / 3)
    train_X, test_X = train_X.iloc[:, 0], test_X.iloc[:, 0]

    go.Figure([go.Scatter(x=X, y=f_X, mode='markers',
                          name="Original Data Without Noise",
                          marker=dict(color="blue")),
               go.Scatter(x=train_X, y=train_y, mode='markers',
                          name="Train Data",
                          marker=dict(color="red")),
               go.Scatter(x=test_X, y=test_y, mode='markers',
                          name="Test Data",
                          marker=dict(color="green"))],
              layout=go.Layout(title="Original Data Without Noise and "
                                     f"Train-Test Split With Noise {noise}",
                               xaxis_title=dict(text="X"),
                               yaxis_title=dict(text="f(X)"))).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = np.zeros(POLYNOMIAL_MAX_DEGREE + 1)
    validation_errors = np.zeros(POLYNOMIAL_MAX_DEGREE + 1)
    all_ks = list(range(POLYNOMIAL_MAX_DEGREE + 1))
    for k in all_ks:
        model = PolynomialFitting(k)
        errors = cross_validate(model, train_X.to_numpy(), train_y.to_numpy(),
                                mean_square_error)
        train_errors[k], validation_errors[k] = errors
    go.Figure([go.Scatter(x=all_ks, y=train_errors, mode='markers+lines',
                          name="Train Errors",
                          marker=dict(color="blue")),
               go.Scatter(x=all_ks, y=validation_errors, mode='markers+lines',
                          name="Validation Errors",
                          marker=dict(color="red"))],
              layout=go.Layout(title="Train Errors & Validation Errors as a "
                                     "Function of Polynomial Degree, "
                                     f"With Noise {noise}",
                               xaxis_title=dict(text="Polynomial Degree"),
                               yaxis_title=dict(text="Error"))).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    optimal_k = np.argmin(validation_errors)
    optimal_model = PolynomialFitting(int(optimal_k))
    optimal_model.fit(train_X.to_numpy(), train_y.to_numpy())
    test_error = optimal_model.loss(test_X.to_numpy(), test_y.to_numpy())
    print(f"With noise {noise}, the optimal polynomial degree was {optimal_k}."
          f"\nIts test error was: {round(test_error, 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X = X[:n_samples]
    train_y = y[:n_samples]
    test_X = X[n_samples:]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    possible_lambda_values = np.linspace(0, 2.5, n_evaluations)
    ridge_train_errors = []
    ridge_validation_errors = []
    lasso_train_errors = []
    lasso_validation_errors = []
    for possible_lambda in possible_lambda_values:
        ridge_errors = cross_validate(RidgeRegression(possible_lambda),
                                      train_X, train_y, mean_square_error)
        ridge_train_errors.append(ridge_errors[0])
        ridge_validation_errors.append(ridge_errors[1])
        lasso_errors = cross_validate(Lasso(possible_lambda),
                                      train_X, train_y, mean_square_error)
        lasso_train_errors.append(lasso_errors[0])
        lasso_validation_errors.append(lasso_errors[1])
    go.Figure([go.Scatter(x=possible_lambda_values, y=ridge_train_errors,
                          mode='markers+lines', name="Train Errors",
                          marker=dict(color="blue")),
               go.Scatter(x=possible_lambda_values, y=ridge_validation_errors,
                          mode='markers+lines', name="Validation Errors",
                          marker=dict(color="red"))],
              layout=go.Layout(title="Ridge Train & Validation Errors as a "
                                     "Function of Regularization's Lambda",
                               xaxis_title=dict(text="Lambda"),
                               yaxis_title=dict(text="Error"))).show()
    go.Figure([go.Scatter(x=possible_lambda_values, y=lasso_train_errors,
                          mode='markers+lines', name="Train Errors",
                          marker=dict(color="blue")),
               go.Scatter(x=possible_lambda_values, y=lasso_validation_errors,
                          mode='markers+lines', name="Validation Errors",
                          marker=dict(color="red"))],
              layout=go.Layout(title="Lasso Train & Validation Errors as a "
                                     "Function of Regularization's Lambda",
                               xaxis_title=dict(text="Lambda"),
                               yaxis_title=dict(text="Error"))).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lam = possible_lambda_values[np.argmin(ridge_validation_errors)]
    best_lasso_lam = possible_lambda_values[np.argmin(lasso_validation_errors)]
    print(f"The lambda value that has achieved best validation error"
          f"for Ridge: {best_ridge_lam}")
    print(f"The lambda value that has achieved best validation error"
          f"for Lasso: {best_lasso_lam}")
    ridge_model = RidgeRegression(best_ridge_lam)
    ridge_model.fit(train_X, train_y)
    ridge_loss = ridge_model.loss(test_X, test_y)
    lasso_model = Lasso(best_lasso_lam)
    lasso_model.fit(train_X, train_y)
    lasso_y_pred = lasso_model.predict(test_X)
    lasso_loss = mean_square_error(test_y, lasso_y_pred)
    least_squares_model = LinearRegression()
    least_squares_model.fit(train_X, train_y)
    least_squares_loss = least_squares_model.loss(test_X, test_y)
    print(f"Test error of Ridge: {ridge_loss}")
    print(f"Test error of Lasso: {lasso_loss}")
    print(f"Test error of Least Squares: {least_squares_loss}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
