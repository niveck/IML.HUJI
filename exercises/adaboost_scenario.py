import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy

Q1_G1 = 'Train Errors'
Q1_G2 = 'Test Errors'
Q1_GRAPH_TITLE = 'Error Rate as a Function of Number of Learners'
Q1_X_TITLE = 'Number of Learners'
Q1_Y_TITLE = 'Error Rate'
Q2_GRAPH_TITLE = 'Decision Surface of 4 Different Numbers of Iterations'
Q2_TITLE_FORMAT = '{0} Iterations'
Q3_TITLE_FORMAT = 'Best Ensemble of Weak Learners - Size: {0}, Accuracy: {1}'
Q4_GRAPH_TITLE = 'Training Set, with Size Proportional to Weights'


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)
    train_errors = []
    test_errors = []
    all_Ts = np.array(range(1, n_learners + 1))
    for T in all_Ts:
        train_errors.append(model.partial_loss(train_X, train_y, T))
        test_errors.append(model.partial_loss(test_X, test_y, T))
    go.Figure([go.Scatter(x=all_Ts, y=train_errors, mode='lines', name=Q1_G1),
               go.Scatter(x=all_Ts, y=test_errors, mode='lines', name=Q1_G2)],
              layout=go.Layout(title=Q1_GRAPH_TITLE, xaxis_title=Q1_X_TITLE,
                               yaxis_title=Q1_Y_TITLE)).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[Q2_TITLE_FORMAT.format(t) for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    symbols = np.array([None, 'x', 'circle'])
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: model.partial_predict(X, t),
                                         lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode='markers', showlegend=False,
                                   marker=dict(color=test_y.astype(int),
                                               symbol=symbols[test_y.
                                                              astype(int)],
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color='black',
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(title=Q2_GRAPH_TITLE, margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    lowest_error, most_accurate_y_hat = None, None
    best_ensemble_size = 0
    for t in all_Ts:
        error_t = model.partial_loss(test_X, test_y, t)
        if lowest_error is None or error_t < lowest_error:
            lowest_error = error_t
            most_accurate_y_hat = model.partial_predict(test_X, t)
            best_ensemble_size = t
    best_accuracy = accuracy(test_y, most_accurate_y_hat)
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda X:
                                     model.partial_predict(X,
                                                           best_ensemble_size),
                                     lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                               mode='markers',
                               showlegend=False,
                               marker=dict(color=test_y.astype(int),
                                           symbol=symbols[test_y.astype(int)],
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color='black',
                                                     width=1)))])
    fig.update_layout(title=Q3_TITLE_FORMAT.format(best_ensemble_size,
                                                   best_accuracy))
    fig.show()

    # Question 4: Decision surface with weighted samples
    normalized_weights = model.D_ / np.max(model.D_) * 25
    fig = go.Figure()
    fig.add_traces([decision_surface(lambda X:
                                     model.partial_predict(X,
                                                           best_ensemble_size),
                                     lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                               mode='markers',
                               showlegend=False,
                               marker=dict(color=train_y.astype(int),
                                           symbol=symbols[train_y.astype(int)],
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color='black',
                                                     width=1),
                                           size=normalized_weights))])
    fig.update_layout(title=Q4_GRAPH_TITLE)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
