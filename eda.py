import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def perform_eda(original_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    print("Shape of data frame", original_df.shape)

    # update pandas config to show all columns when display data set example
    pd.pandas.set_option('display.max_columns', None)

    # display data set example
    print(original_df.head())

    # display summary of dataframe
    print(original_df.info())

    # replace all "?" with null value
    df = original_df.replace('?', np.nan)

    # generate bar charts based on categorical attributes for visualization
    categorical_attributes_plot(df)

    # generate bar histograms on numerical attributes for visualization
    numerical_attributes_plot(df)

    # count the number of null value in each attribute
    null_sum_series = df.isnull().sum()

    # filter out the features with null value
    null_series = null_sum_series[null_sum_series > 0]
    # display the attributes with null value
    print(null_series)

    missing_value_features = null_series.keys()
    for feature in missing_value_features:
        # since the attributes with missing value are all categorical data
        # perform most frequent imputation, replace missing value with mode
        mode = df[feature].mode()[0]
        df[feature].fillna(mode, inplace=True)

    # verify if all null or missing value are replaced
    print(df.isnull().sum())

    # drop the label column
    features = df.drop(['income'], axis=1)

    label = df['income']
    return features, label


def categorical_attributes_plot(df: pd.DataFrame):
    # filter by non-number data type which will be categorical attributes
    fields = df.select_dtypes(exclude="number").columns

    cols = 3
    rows = math.ceil(len(fields) / cols)

    plt.subplots(rows, cols, figsize=(20, 14))
    for i in range(1, len(fields) + 1):
        plt.subplot(rows, cols, i)
        df[fields[i - 1]].value_counts().sort_values().plot.bar()
        plt.xticks(rotation=90)
        plt.title(fields[i - 1])

    plt.tight_layout()
    plt.savefig("./figures/categorical_attributes_bar_chart.png")
    plt.show()


def numerical_attributes_plot(df: pd.DataFrame):
    # filter by number data type which will be numerical attributes
    fields = df.select_dtypes(include="number").columns

    cols = 3
    rows = math.ceil(len(fields) / cols)

    df.hist(bins=20, figsize=(20, 14), layout=(rows, cols))

    plt.tight_layout()
    plt.savefig("./figures/numerical_attributes_histogram.png")
    plt.show()
