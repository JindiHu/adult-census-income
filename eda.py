import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# to ignore the future warnings from seaborn library
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

categorical_cols = ["workclass", "education", "marital-status", "occupation",
                    "relationship", "race", "sex", "native-country"]
numerical_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def perform_eda(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    print("Shape of data frame", df.shape)

    # update pandas config to show all columns when display data set example
    pd.pandas.set_option("display.max_columns", None)

    # display data set example
    print(df.head())
    print("*" * 10, "\n")

    # display summary of dataframe
    print(df.info())
    print("*" * 10, "\n")

    # generate bar charts based on categorical attributes for visualization
    categorical_attributes_plot(df)

    # generate bar histograms on numerical attributes for visualization
    numerical_attributes_plot(df)

    replace_missing_value(df)

    remove_outliers(df)

    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    plot_heatmap(df)

    feature_engineering(df)

    print(df.head())

    encode_values(df)

    x, y = split_x_y(df)

    return x, y


def categorical_attributes_plot(df: pd.DataFrame):
    columns = categorical_cols + ["income"]
    cols = 3
    rows = math.ceil(len(categorical_cols) / cols)

    plt.subplots(rows, cols, figsize=(20, 14))
    for i in range(len(columns)):
        plt.subplot(rows, cols, i + 1)
        df[columns[i]].value_counts().sort_values().plot.bar()
        plt.xticks(rotation=90)
        plt.title(columns[i])

    plt.tight_layout()
    plt.savefig("./figures/categorical_attributes_bar_chart.png")
    plt.show()

    plt.subplots(rows, cols, figsize=(30, 20))
    for i in range(len(columns)):
        plt.subplot(rows, cols, i + 1)
        attribute = columns[i]
        low_income = df.loc[df["income"] == "<=50K", attribute]
        high_income = df.loc[df["income"] == ">50K", attribute]
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
        plt.xlabel("Number of Entries")
        plt.legend([low_bar, high_bar], ["<=50K", ">50K"])
        plt.yticks(np.arange(len(high_income_stats)), high_income_stats.index)
    plt.tight_layout()
    plt.savefig("./figures/categorical_attributes_vs_income_bar_chart.png")
    plt.show()


def numerical_attributes_plot(df: pd.DataFrame):
    cols = 3
    rows = math.ceil(len(numerical_cols) / cols)

    df.hist(bins=20, figsize=(20, 14), layout=(rows, cols))
    plt.tight_layout()
    plt.savefig("./figures/numerical_attributes_histogram.png")
    plt.show()

    plt.subplots(rows, cols, figsize=(20, 14))
    for i in range(len(numerical_cols)):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=df[numerical_cols[i]])
    plt.tight_layout()
    plt.savefig("./figures/numerical_attributes_boxplot.png")
    plt.show()


def plot_heatmap(df: pd.DataFrame):
    corr = df.select_dtypes("number").corr()
    sns.heatmap(corr, annot=True, fmt=".3f")
    plt.tight_layout()
    plt.savefig("./figures/numerical_attributes_correlation_heatmap.png")
    plt.show()


def replace_missing_value(df: pd.DataFrame):
    # replace all "?" with null value
    df = df.replace("?", np.nan)

    # count the number of null value in each attribute
    null_sum_series = df.isnull().sum()

    # filter out the features with null value
    null_series = null_sum_series[null_sum_series > 0]
    # display the attributes with null value

    missing_value_features = null_series.keys()
    for feature in missing_value_features:
        # since the attributes with missing value are all categorical data
        # perform most frequent imputation, replace missing value with mode
        mode = df[feature].mode()[0]
        df[feature].fillna(mode, inplace=True)


def remove_outliers(df: pd.DataFrame):
    cols_with_outliers = numerical_cols
    for i in range(len(cols_with_outliers)):
        attribute = df[cols_with_outliers[i]]
        upper = attribute.quantile(0.75) + 1.5 * (attribute.quantile(0.75) - attribute.quantile(0.25))
        lower = attribute.quantile(0.25) - 1.5 * (attribute.quantile(0.75) - attribute.quantile(0.25))
        print("column -> ", cols_with_outliers[i], "")
        x = df[(attribute < round(lower, 2)) | (attribute > round(upper, 2))][cols_with_outliers[i]]
        print("no of outliers present -> ", len(x))
        print("*" * 10, "\n")
        df = df[(df[cols_with_outliers[i]] >= lower) & (df[cols_with_outliers[i]] <= upper)]
        print("data shape after removing outliers of", cols_with_outliers[i], ":", df.shape)
        print("*" * 10, "\n")


def feature_engineering(df: pd.DataFrame):
    # combine values for some of the categorical attributes
    df["education"].replace(["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th"],
                            "School", inplace=True)
    df["education"].replace(["Some-college", "Assoc-acdm", "Assoc-voc"], "College", inplace=True)
    df["marital-status"].replace(["Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent"],
                                 "Married", inplace=True)
    df["native-country"].replace(["United-States", "Canada", "Outlying-US(Guam-USVI-etc)", "Haiti",
                                  "Nicaragua"],
                                 "North-America",
                                 inplace=True)
    df["native-country"].replace(["Puerto-Rico", "Cuba", "Honduras", "Jamaica", "Mexico",
                                  "Dominican-Republic", "Guatemala", "El-Salvador", "Trinadad&Tobago"],
                                 "Central-America",
                                 inplace=True)
    df["native-country"].replace(["Ecuador", "Columbia", "Peru"], "South-America",
                                 inplace=True)
    df["native-country"].replace(["Cambodia", "India", "Japan", "South", "China", "Philippines",
                                  "Vietnam", "Laos", "Taiwan", "Thailand", "Hong", ],
                                 "Asia",
                                 inplace=True)
    df["native-country"].replace(["England", "Germany", "Greece", "Italy", "Poland", "Portugal", "Ireland",
                                  "France", "Hungary", "Scotland", "Yugoslavia", "Holand-Netherlands"],
                                 "Europe",
                                 inplace=True)

    # df["native-country"] = df["native-country"].where(df["native-country"] == "United-States", "Others")
    df["race"].replace(["Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], " Others", inplace=True)

    df["age"] = pd.qcut(df["age"], q=4, duplicates="drop")
    df["hours-per-week"] = pd.cut(df["hours-per-week"], bins=5)
    df["education-num"] = pd.qcut(df["education-num"], q=4, duplicates="drop")
    df["net-gain"] = df["capital-gain"] - df["capital-loss"]
    df["net-gain"] = pd.qcut(df["net-gain"], q=4, duplicates="drop")
    df["fnlwgt"] = pd.qcut(df["fnlwgt"], q=4, duplicates="drop")

    df.drop(["capital-gain", "capital-loss"], axis=1, inplace=True)


def encode_values(df: pd.DataFrame):
    label_encoder = LabelEncoder()
    for col in df:
        label_encoder.fit(df[col])
        df[col] = label_encoder.transform(df[col])


def split_x_y(df: pd.DataFrame):
    # drop the label column
    x = df.drop(["income"], axis=1)

    y = df["income"]

    return x, y
