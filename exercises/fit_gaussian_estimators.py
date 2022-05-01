from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"

Q1_mu = 10
Q1_sigma = 1
Q1_NUM_SAMPLES = 1000
Q1_ANS_FORMAT = "({expectation}, {variance})"
Q2_START_RANGE = 10
Q2_END_RANGE = 1001
Q2_JUMP = 10
Q2_GRAPH_TITLE = r"$\text{Absolute Distance Between the Estimated and True Value of " \
                 r"the Expectation, as a Function of the Sample Size}$"
Q2_X_TITLE = r"$\text{Sample Size}$"
Q2_Y_TITLE = r"$\text{Absolute Distance Between the Estimated and True Value " \
             r"of the Expectation}$"
Q3_GRAPH_TITLE = r"$\text{Empirical PDF Function Under the Fitted Model}$"
Q3_X_TITLE = r"$\text{Samples}$"
Q3_Y_TITLE = r"$\text{Empirical PDF}$"
Q4_MU = np.array([0, 0, 4, 0])
Q4_COV = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0],
                   [0, 0, 1, 0], [0.5, 0, 0, 1]])
Q4_NUM_SAMPLES = 1000
Q4_MSG = "Estimated expectation: {mu}\nEstimated Covariance:\n{cov}"
Q5_GRAPH_TITLE = r"$\text{Heatmap of the Log Likelihood for }" \
                 r"f_1\text{ and }f_3$"
Q5_X_TITLE = r"$f_3$"
Q5_Y_TITLE = r"$f_1$"
Q6_MSG = "The model which achieved the maximum log-likelihood value is:\n" \
         "f1 = {f1}, f3 = {f3}"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(Q1_mu, Q1_sigma, Q1_NUM_SAMPLES)
    q1_model = UnivariateGaussian()
    q1_model.fit(samples)
    print(Q1_ANS_FORMAT.format(expectation=q1_model.mu_,
                               variance=q1_model.var_))

    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = [i for i in range(Q2_START_RANGE, Q2_END_RANGE, Q2_JUMP)]
    abs_distances = []
    for size in sample_sizes:
        q2_new_model = UnivariateGaussian()
        q2_new_model.fit(samples[:size])
        abs_distances.append(np.abs(Q1_mu - q2_new_model.mu_))
    go.Figure([go.Scatter(x=sample_sizes, y=abs_distances, mode='markers')],
              layout=go.Layout(title=Q2_GRAPH_TITLE, xaxis_title=Q2_X_TITLE,
                               yaxis_title=Q2_Y_TITLE)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = q1_model.pdf(samples)
    go.Figure([go.Scatter(x=samples, y=pdf, mode='markers')],
              layout=go.Layout(title=Q3_GRAPH_TITLE, xaxis_title=Q3_X_TITLE,
                               yaxis_title=Q3_Y_TITLE)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samples = np.random.multivariate_normal(Q4_MU, Q4_COV, Q4_NUM_SAMPLES)
    q4_model = MultivariateGaussian()
    q4_model.fit(samples)
    print(Q4_MSG.format(mu=q4_model.mu_, cov=q4_model.cov_))

    # Question 5 - Likelihood evaluation
    f_values = np.linspace(-10, 10, 200)
    log_likelihood = []
    for f1 in f_values:
        log_likelihood_by_f1 = []
        for f3 in f_values:
            mu = np.array([f1, 0, f3, 0])
            q5_new_model = MultivariateGaussian()
            log_likelihood_by_f1.append(q5_new_model.log_likelihood(mu, Q4_COV,
                                                                    samples))
        log_likelihood.append(log_likelihood_by_f1)
    go.Figure(go.Heatmap(x=f_values, y=f_values, z=log_likelihood),
              layout=go.Layout(title=Q5_GRAPH_TITLE, xaxis_title=Q5_X_TITLE,
                               yaxis_title=Q5_Y_TITLE)).show()

    # # Question 6 - Maximum likelihood
    max_f_indices = np.where(log_likelihood == np.max(log_likelihood))
    max_f1_i, max_f3_i = max_f_indices[0], max_f_indices[1]
    max_f1 = round(f_values[max_f1_i][0], 3)
    max_f3 = round(f_values[max_f3_i][0], 3)
    print(Q6_MSG.format(f1=max_f1, f3=max_f3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
