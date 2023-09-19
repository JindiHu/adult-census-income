import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import constant


def perform_eda(original_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    print("Shape of data frame", original_df.shape)

    # update pandas config to show all columns when display data set example
    pd.pandas.set_option('display.max_columns', None)

    # display data set example
    print(original_df.head())
    print("*" * 10, "\n")

    # display summary of dataframe
    print(original_df.info())
    print("*" * 10, "\n")

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
    print("*" * 10, "\n")

    missing_value_features = null_series.keys()
    for feature in missing_value_features:
        # since the attributes with missing value are all categorical data
        # perform most frequent imputation, replace missing value with mode
        mode = df[feature].mode()[0]
        df[feature].fillna(mode, inplace=True)

    remove_outliers(df)

    # verify if all null or missing value are replaced
    print(df.isnull().sum())
    print("*" * 10, "\n")

    # create new attribute capital net gain (profile)
    df["capital-net"] = df["capital-gain"] - df["capital-loss"]

    # drop capital-gain and capital-gain from dataframe
    df.drop(["capital-gain", "capital-loss"], axis="columns", inplace=True)

    # drop the label column
    x = df.drop(["income"], axis="columns")

    print(x.head())
    print("*" * 10, "\n")

    y = df["income"]

    print(y.head())
    print("*" * 10, "\n")
    return x, y


def categorical_attributes_plot(df: pd.DataFrame):
    # filter by non-number data type which will be categorical attributes
    fields = df.select_dtypes(exclude="number").columns

    cols = 3
    rows = math.ceil(len(fields) / cols)

    plt.subplots(rows, cols, figsize=(20, 14))
    for i in range(len(fields)):
        plt.subplot(rows, cols, i + 1)
        df[fields[i]].value_counts().sort_values().plot.bar()
        plt.xticks(rotation=90)
        plt.title(fields[i])

    plt.tight_layout()
    plt.savefig("./figures/categorical_attributes_bar_chart.png")
    plt.show()

    plt.subplots(rows, cols, figsize=(30, 20))
    for i in range(len(fields)):
        plt.subplot(rows, cols, i + 1)
        attribute = fields[i]
        low_income = df.loc[df[constant.TARGET] == '<=50K', attribute]
        high_income = df.loc[df[constant.TARGET] == '>50K', attribute]
        low_income_stats = low_income.value_counts()
        high_income_stats = high_income.value_counts()

        low_bar = plt.barh(
            low_income_stats.index,
            low_income_stats.values,
            alpha=0.5,
        )
        high_bar = plt.barh(
            high_income_stats.index,
            high_income_stats.values,
            alpha=0.5,
        )

        plt.title(attribute)
        plt.xlabel('Number of Entries')
        plt.legend([low_bar, high_bar], ['<=50K', '>50K'])
        plt.yticks(np.arange(len(high_income_stats)), high_income_stats.index)
    plt.tight_layout()
    plt.savefig("./figures/categorical_attributes_vs_income_bar_chart.png")
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

    plt.subplots(rows, cols, figsize=(20, 14))
    for i in range(len(fields)):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=df[fields[i]])
    plt.tight_layout()
    plt.savefig("./figures/numerical_attributes_boxplot.png")
    plt.show()

    # calculate the correlation matrix on the numeric columns
    corr = df.select_dtypes('number').corr()

    # plot the heatmap
    sns.heatmap(corr, annot=True, fmt=".3f")
    plt.tight_layout()
    plt.savefig("./figures/numerical_attributes_correlation_heatmap.png")
    plt.show()


def remove_outliers(df: pd.DataFrame):
    fields = df.select_dtypes(include="number").columns
    for i in range(len(fields)):
        attribute = df[fields[i]]
        upper = attribute.quantile(0.75) + 1.5 * (attribute.quantile(0.75) - attribute.quantile(0.25))
        lower = attribute.quantile(0.25) - 1.5 * (attribute.quantile(0.75) - attribute.quantile(0.25))
        print("column -> ", fields[i], "")
        x = df[(attribute < round(lower, 2)) | (attribute > round(upper, 2))][fields[i]]
        print("No of Outliers present -> ", len(x))
        print("*" * 10, '\n')
        df = df[(df[fields[i]] >= lower) & (df[fields[i]] <= upper)]
        print("data shape after removing outliers of ", fields[i], ":", df.shape)
        print("*" * 10, "\n")
