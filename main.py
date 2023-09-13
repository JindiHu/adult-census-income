import sys
import pandas as pd
import constant


def load_data(file_name: str):
    data = pd.read_csv(file_name, names=constant.DATA_NAMES, delimiter=",")
    return data


if __name__ == '__main__':
    if sys.version_info[0:2] != (3, 11):
        raise Exception('Requires python 3.11')

    load_data(constant.TRAIN_DATASET_FILEPATH)

