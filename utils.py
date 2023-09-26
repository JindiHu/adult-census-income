import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(file_name: str, column_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_name, names=column_names, delimiter=",")

    # trim the whitespace for each column
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return df


def encode_values(cols: list, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
    label_encoder = LabelEncoder()
    for col in cols:
        label_encoder.fit(train_df[col])
        train_df[col] = label_encoder.transform(train_df[col])
        if test_df is not None:
            test_df[col] = label_encoder.transform(test_df[col])
