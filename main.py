import sys

from sklearn.model_selection import train_test_split

import utils
from decision_tree import DecisionTree
from eda import perform_eda
from random_forest import RandomForest

col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
             "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
             "hours-per-week", "native-country", "income"]

if __name__ == "__main__":
    if sys.version_info[0:2] != (3, 11):
        raise Exception("Requires python 3.11")

    # load dataset into pandas dataframe
    df = utils.load_data("./census_income/adult.data", col_names)

    # perform Exploratory Data Analysis (EDA)
    x, y = perform_eda(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    dt = DecisionTree(features=x_train, target=y_train, max_depth=6)
    print("training decision tree ...")
    dt.fit()
    print("evaluating decision tree ...")
    dt.evaluate(x_test, y_test)

    rf = RandomForest(features=x_train, target=y_train, n_estimators=6, sample_frac=0.8, max_depth=4)
    print("training random forest ...")
    rf.fit()
    print("evaluating random forest ...")
    rf.evaluate(x_test, y_test)
