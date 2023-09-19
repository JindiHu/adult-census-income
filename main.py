import sys

import constant
import utils
from eda import perform_eda

if __name__ == '__main__':
    if sys.version_info[0:2] != (3, 11):
        raise Exception('Requires python 3.11')

    # load dataset into pandas dataframe
    df = utils.load_data(constant.TRAIN_DATASET_FILEPATH, constant.DATA_NAMES)

    # perform Exploratory Data Analysis (EDA)
    df = perform_eda(df)
