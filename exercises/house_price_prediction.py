from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

DATA_PATH = r"..\datasets\house_prices.csv"
Q2_ANS_PATH = r"C:\Users\nivec\HUJI Drive\Year 2\Semester B\IML\ex2"
Q4_GRAPH_TITLE = 'The Mean Loss as a Function of Percentage of Train Set Size'
Q4_X_TITLE = {'text': 'Percentage of Train Set Size'}
Q4_Y_TITLE = {'text': 'Mean Loss (with confidence interval)'}


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.fillna(0)
    columns_with_positive_values = ['id', 'price', 'bedrooms', 'bathrooms',
                                    'sqft_lot15']
    # to remove all samples with 0 >= in them
    for col in columns_with_positive_values:
        df = df[df[col] > 0]
    # specific columns problems:
    df = df[(df['bedrooms'] <= 11) & (df['date'] != '0')]
    df['date'] = pd.to_datetime(df['date'])
    df['zipcode'] = df['zipcode'].astype(int)
    # new columns:
    df['years_since_built'] = df['date'].dt.year - df['yr_built']
    df['years_since_last_touch'] = df['date'].dt.year - df[
        ["yr_built", "yr_renovated"]].max(axis=1)
    df = df[(df['years_since_built'] > 0) & (df['years_since_last_touch'] > 0)]
    columns_to_dummify = ['zipcode']
    for col in columns_to_dummify:
        df = pd.get_dummies(df, columns=[col], prefix=col)
    prices = df['price']
    columns_to_drop = ['id', 'date', 'yr_renovated', 'yr_built', 'price',
                       'lat', 'long']
    df.drop(columns_to_drop, axis=1, inplace=True)
    return df, prices


def pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    """
    x: array of floats
    y: array of floats
    return: Pearson's Correlation between x and y
    """
    return x.cov(y) / (x.std() * y.std())


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col_title in X:
        corr = pearson_correlation(X[col_title], y)
        title = f'"{y.name}" as function of "{col_title}", Correlation: {corr}'
        pic_name = f'{y.name} as func of {col_title} corr {corr}.png'
        graph = px.scatter(pd.concat([X[col_title], y], axis=1), x=col_title,
                           y=y.name, title=title)
        graph.write_image(Q2_ANS_PATH + '\\' + pic_name)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(DATA_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    training_percentage = [i for i in range(10, 101)]
    mean_error_per_percent = []
    std_error_per_percent = []
    for p in training_percentage:
        errors = []
        for i in range(10):
            model = LinearRegression()
            cur_train_X, cur_train_y, _, _\
                = split_train_test(train_X, train_y, train_proportion=p/100)
            model.fit(cur_train_X.to_numpy(), cur_train_y.to_numpy())
            loss = model.loss(test_X.to_numpy(), test_y.to_numpy())
            errors.append(loss)
        mean_error_per_percent.append(np.mean(errors))
        std_error_per_percent.append(np.std(errors))
    confidence_interval_up_bound = [mean_error_per_percent[i] +
                                    2 * std_error_per_percent[i]
                                    for i in range(len(training_percentage))]
    confidence_interval_low_bound = [mean_error_per_percent[i] -
                                     2 * std_error_per_percent[i]
                                     for i in range(len(training_percentage))]
    go.Figure([go.Scatter(x=training_percentage, y=mean_error_per_percent,
                          mode='markers+lines', name="Mean Loss"),
               go.Scatter(x=training_percentage,
                          y=confidence_interval_low_bound, fill=None,
                          mode="lines", line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=training_percentage,
                          y=confidence_interval_up_bound, fill='tonexty',
                          mode="lines", line=dict(color="lightgrey"),
                          showlegend=False)],
              layout=go.Layout(title=Q4_GRAPH_TITLE, xaxis_title=Q4_X_TITLE,
                               yaxis_title=Q4_Y_TITLE)).show()
