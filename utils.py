import pandas as pd


def load_data(file_name: str, column_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_name, names=column_names, delimiter=",")

    # trim the whitespace for each column
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    return df
