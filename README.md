# Overview

This project is part of the QRT Challenge, where the goal is to predict stock price movements using machine learning and quantitative research techniques, specifically *signature analysis* and *transfer learning*; novel ideas published in Prof. Xin Guo's research papers.
The challenge involves processing financial data, engineering features, and developing predictive models to identify patterns in stock price behavior.
Key aspects of the project include:

(1) Data Cleansing: Abnormal value (Nan, None) detection, Aggregating market data, identifying issues, disproportionate sectors, etc

(2) Feature Engineering: Extracting meaningful features that capture market dynamics.

(3) Model Training/Optimization: Implementing and optimizing machine learning models for price movement prediction, based on *signature analysis*.

(4) Evaluation: Assessing model performance using backtesting and statistical metrics.

## Dataset

After creating an account, you can download the dataset from the following [website](https://challengedata.ens.fr/challenges/143)
Unzip the file, and there will be 4 csv files.
Create a folder called `datasets` at the root level of the directory, and then drag the data inside. We cannot upload the data to Github because of the size constraint.

## Data Cleaning/Signature Computation

This project heavily revolves on the computation of signatures, so we have included both data preprocessing as well as signature computation in `calc_signature.py`. We can then use this function to clean/process our datasets before running our regression models.

Note: Cannot currently save the dataframe easily, because of how large it is in size.

## Regression

We can run the regression model using `ridge_regression_by_sector`; the `.ipynb` and `.py` are the same just different speeds.

The `lgbm_model.py` is a different ML model selection that we are testing for now, but primarily we will be using the ridge regression.

Currently we are testing `Lasso Regression` in comparison with `Ridge regression` to see how the prediction accuracy will change accordingly.