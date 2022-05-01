import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

DATA_PATH = r"..\datasets\City_Temperature.csv"
Q2_GRAPH_TITLE1 = 'Temperature in Israel as a Function of Day of Year'
Q2_GRAPH_TITLE2 = 'Temperature\'s Standard Deviation in Israel by Month'
Q3_GRAPH_TITLE = 'Mean Monthly Temperature by Country, with Error Bars'
Q4_GRAPH_TITLE = 'Loss Value as a Function of Degree of the Polynomial Model'
Q4_X_TITLE = 'Degree of the Polynomial Model'
Q4_Y_TITLE = 'Loss Value'
CHOSEN_DEGREE = 5
COUNTRIES = ['Jordan', 'South Africa', 'The Netherlands']
Q5_GRAPH_TITLE = 'Loss Value of the Model as a Function of Country'
Q5_X_TITLE = 'Country'
Q5_Y_TITLE = 'Loss Value'


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear
    # remove lines with invalid data:
    df = df[df['Temp'] > -20]  # using reasonable temp threshold for NL
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(DATA_PATH)

    # Question 2 - Exploring data for specific country
    israel_temp_data = data[data['Country'] == 'Israel']
    # treat Year values scale as discrete instead of continuous:
    israel_temp_data = israel_temp_data.astype({'Year': str})
    px.scatter(israel_temp_data, x='DayOfYear', y='Temp', color='Year',
               title=Q2_GRAPH_TITLE1).show()
    px.bar(israel_temp_data.groupby('Month').agg({'Temp': 'std'}),
           title=Q2_GRAPH_TITLE2).show()

    # Question 3 - Exploring differences between countries
    px.line(data.groupby(['Country', 'Month'])
            .agg(Std=('Temp', 'std'), Mean=('Temp', 'mean')).reset_index(),
            x='Month', y='Mean', error_y='Std', color='Country',
            title=Q3_GRAPH_TITLE).show()

    # Question 4 - Fitting model for different values of `k`
    X, y = israel_temp_data['DayOfYear'], israel_temp_data['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    errors = []
    degrees = list(range(1, 11))
    for k in degrees:
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = round(model.loss(test_X.to_numpy(), test_y.to_numpy()), 2)
        errors.append(loss)
        print(f'Polynomial Model Degree (k): {k}, Loss Value: {loss}')
    px.bar(x=degrees, y=errors, labels={'x': Q4_X_TITLE, 'y': Q4_Y_TITLE},
           title=Q4_GRAPH_TITLE).show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(CHOSEN_DEGREE)
    model.fit(israel_temp_data['DayOfYear'].to_numpy(),
              israel_temp_data['Temp'].to_numpy())
    errors = []
    countries = COUNTRIES
    for country in countries:
        test_X = data[data['Country'] == country]['DayOfYear']
        test_y = data[data['Country'] == country]['Temp']
        errors.append(model.loss(test_X, test_y))
    px.bar(x=countries, y=errors, labels={'x': Q5_X_TITLE, 'y': Q5_Y_TITLE},
           title=Q5_GRAPH_TITLE).show()
