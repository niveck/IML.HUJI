from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

PATH_PREFIX = "../datasets/"
LOSS_ITER_TITLE = "Loss as a Function of Fitting Iteration for "
ITER_AXIS_TITLE = "Fitting Iteration Number"
LOSS_AXIS_TITLE = "Loss Value"
GNB_ACCURACY_TITLE = "Gaussian Naive Bayes Model's Accuracy: "
LDA_ACCURACY_TITLE = "LDA Model's Accuracy: "
COMPARISON_TITLE = "Prediction of Two Models Over Dataset: "
COLORS = np.array(["red", "blue", "green"])
SYMBOLS = np.array(["circle", "triangle-ne", "star-open"])


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(PATH_PREFIX + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_recording(fit, X_i, y_i): losses.append(fit.loss(X, y))

        model = Perceptron(callback=loss_recording)
        model.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure([go.Scatter(x=[i + 1 for i in range(len(losses))], y=losses,
                              mode='markers')],
                  layout=go.Layout(title=LOSS_ITER_TITLE + n,
                                   xaxis_title=ITER_AXIS_TITLE,
                                   yaxis_title=LOSS_AXIS_TITLE)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(PATH_PREFIX + f)

        # Fit models and predict over training set
        gnb_model = GaussianNaiveBayes()
        gnb_model.fit(X, y)
        y_pred_gnb = gnb_model.predict(X)

        lda_model = LDA()
        lda_model.fit(X, y)
        y_pred_lda = lda_model.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        gnb_accuracy = accuracy(y, y_pred_gnb)
        lda_accuracy = accuracy(y, y_pred_lda)
        figure = make_subplots(rows=1, cols=2, subplot_titles=[
            GNB_ACCURACY_TITLE + str(gnb_accuracy),
            LDA_ACCURACY_TITLE + str(lda_accuracy)])

        # Add traces for data-points setting symbols and colors
        gnb_graph = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                               marker=dict(color=COLORS[y_pred_gnb],
                                           symbol=SYMBOLS[y]),
                               showlegend=False)
        figure.add_trace(gnb_graph, row=1, col=1)
        lda_graph = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                               marker=dict(color=COLORS[y_pred_lda],
                                           symbol=SYMBOLS[y]),
                               showlegend=False)
        figure.add_trace(lda_graph, row=1, col=2)
        figure.update_layout(title=COMPARISON_TITLE + f, margin=dict(t=75))

        # Add `X` dots specifying fitted Gaussians' means
        gnb_fitted_center = go.Scatter(x=gnb_model.mu_[:, 0],
                                       y=gnb_model.mu_[:, 1], mode="markers",
                                       marker=dict(color="black", symbol="x"),
                                       showlegend=False)
        figure.add_trace(gnb_fitted_center, row=1, col=1)
        lda_fitted_center = go.Scatter(x=lda_model.mu_[:, 0],
                                       y=lda_model.mu_[:, 1], mode="markers",
                                       marker=dict(color="black", symbol="x"),
                                       showlegend=False)
        figure.add_trace(lda_fitted_center, row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(gnb_model.classes_.size):
            gnb_cov_ellipse = get_ellipse(gnb_model.mu_[k],
                                          np.diag(gnb_model.vars_[k]))
            gnb_cov_ellipse.showlegend = False
            figure.add_trace(gnb_cov_ellipse, row=1, col=1)
            lda_cov_ellipse = get_ellipse(lda_model.mu_[k], lda_model.cov_)
            lda_cov_ellipse.showlegend = False
            figure.add_trace(lda_cov_ellipse, row=1, col=2)

        figure.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
