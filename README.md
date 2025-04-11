# Overview

This project is part of the QRT Challenge, where the goal is to predict stock price movements using machine learning and quantitative research techniques, specifically *signature analysis* and *transfer learning*; novel ideas published in Prof. Xin Guo's research papers.
The challenge involves processing financial data, engineering features, and developing predictive models to identify patterns in stock price behavior.
Key aspects of the project include:

(1) Data Cleansing: Abnormal value (Nan, None) detection, Aggregating market data, identifying issues, disproportionate sectors, etc

(2) Feature Engineering: Extracting meaningful features that capture market dynamics.

(3) Model Training/Optimization: Implementing and optimizing machine learning models for price movement prediction, based on *signature analysis*.

(4) Evaluation: Assessing model performance using backtesting and statistical metrics.

## Dataset
Create a folder called `datasets` at the root level of the directory, and then drag the data inside. We cannot upload the data to Github because of the size constraint. The data is from Yahoo Finance.

## Data Cleaning/Signature Computation

This project heavily revolves on the computation of signatures, so we have included both data preprocessing/download `data_cleansing.py`. NOTE: THIS WILL BE UPDATED BY @JAMES. For now, we can just drag the dataset from James into the datasets folder.

1) We first will preprocess the data and calculate the signatures by using `calc_signature.py`.

At this point, our data is already preprocessed and we can run regressions/clustering. 

To directly run the ridge regression using the given stock features in the dataset by sector/industry, we can use `ridgeRegression_groupby.py`

To create automatic clusters using kmeans and clustering, we can run `sig_kernel_adaptive_weight.py` and `sig_kernel_clustering.py`, then after that run the `ridgeRegression_groupby.py`

Finally, to generate any results, visualizations, data exploration we can use `analyze_ridge_regression_experiements.ipynb`
