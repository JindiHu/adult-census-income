import sys
import time

from matplotlib import pyplot as plt

import utils
from decision_tree import DecisionTree
from eda import perform_eda, replace_missing_value, feature_engineering, split_x_y
from random_forest import RandomForest

col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
             "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
             "hours-per-week", "native-country", "income"]


def plot_performance_and_speed(x_label, x_list, x_ticks, accuracy_list, training_time_list, file_name):
    plt.subplots(nrows=1, ncols=2, figsize=(20, 14))
    plt.subplot(1, 2, 1)
    plt.plot(x_list, accuracy_list)
    plt.title('Accuracy vs ' + x_label)
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(x_list, training_time_list)
    plt.title('Training Time vs ' + x_label)
    plt.xlabel(x_label)
    plt.xticks(x_ticks)
    plt.ylabel('Training Time')
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


if __name__ == "__main__":
    if sys.version_info[0:2] != (3, 11):
        raise Exception("Requires python 3.11")

    # load training dataset into pandas dataframe
    df = utils.load_data("./census_income/adult.data", col_names)

    # perform Exploratory Data Analysis (EDA)
    df = perform_eda(df)

    # load test dataset into pandas dataframe
    df_test = utils.load_data("./census_income/adult.test", col_names)
    replace_missing_value(df_test)
    df_test = feature_engineering(df_test)
    df_test["income"] = df_test["income"].map({"<=50K.": 0, ">50K.": 1})

    # encode both train dataset and test dataset
    utils.encode_values(cols=["workclass", "education", "marital-status", "occupation",
                              "relationship", "race", "sex", "native-country"], train_df=df, test_df=df_test)

    x_train, y_train = split_x_y(df)
    x_test, y_test = split_x_y(df_test)

    dt_max_depth = [0]
    dt_accuracy = [0]
    dt_training_time = [0]
    for max_depth in range(1, 21):
        dt = DecisionTree(features=x_train, target=y_train, max_depth=max_depth)
        print("training DecisionTree( max_depth=", max_depth, ") ...")
        start = time.time()
        dt.fit()
        end = time.time()
        elapsed_time = end - start
        print("elapsed time for training:", elapsed_time, "sec")
        print("evaluating DecisionTree( max_depth=", max_depth, ") ...")
        accuracy = dt.evaluate(x_test, y_test)
        dt_max_depth.append(max_depth)
        dt_accuracy.append(accuracy)
        dt_training_time.append(elapsed_time)

    plot_performance_and_speed(x_label="Max Depth", x_list=dt_max_depth, accuracy_list=dt_accuracy,
                               training_time_list=dt_training_time, x_ticks=range(1, 21),
                               file_name="./figures/decision_tree_performance_&_speed.png")

    rf_n_estimators = [0]
    rf_accuracy = [0]
    rf_training_time = [0]
    sample_frac = 0.1
    max_depth = 5
    for n_estimators in range(1, 51):
        rf = RandomForest(features=x_train, target=y_train, n_estimators=n_estimators,
                          sample_frac=sample_frac, max_depth=max_depth)
        print("training RandomForest( n_estimators=", n_estimators, ", sample_frac=",
              sample_frac, ", max_depth=", max_depth, ") ...")
        start = time.time()
        rf.fit()
        end = time.time()
        elapsed_time = end - start
        print("elapsed time for training:", elapsed_time, "sec")
        print("evaluating RandomForest( n_estimators=", n_estimators, ", sample_frac=",
              sample_frac, ", max_depth=", max_depth, ") ...")
        accuracy = rf.evaluate(x_test, y_test)

        rf_n_estimators.append(n_estimators)
        rf_accuracy.append(accuracy)
        rf_training_time.append(elapsed_time)

    plot_performance_and_speed(x_label="Num of Estimators", x_list=rf_n_estimators, accuracy_list=rf_accuracy,
                               training_time_list=rf_training_time, x_ticks=range(1, 51),
                               file_name="./figures/random_forest_performance_&_speed_against_n_estimators.png")
