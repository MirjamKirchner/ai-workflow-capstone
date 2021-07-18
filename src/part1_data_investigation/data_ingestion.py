import cslib as cs
import pandas as pd
from typing import Tuple


def _merge_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    x, y, dates = cs.engineer_features(df, training=False)
    x["date"] = dates
    x["next_months_revenue"] = y
    df_processed = df.merge(x, on="date", how="left")

    return df_processed


def ingest(dir_raw: str, path_processed: str, append: bool = False) -> pd.DataFrame:
    """
    Reads all raw data in JSON-format, processes it to a data frame, and stores as it a CSV-file in a separate directory
    :param dir_raw: directory to the raw data
    :param path_processed: path to the processed input data in CSV-formate
    :param append: if path_processed already exists, the data will be appended to the existing file will be replaced if
    True, otherwise the file will be overwritten
    :return:
    """
    df_raw = cs.fetch_data(dir_raw)
    countries = df_raw["country"].unique()

    # Convert to time series
    dfs_ts = {country: cs.convert_to_ts(df_raw, country=country) for country in countries}

    # Add some additional features
    dfs_processed = [_merge_engineered_features(df).assign(country=country) for country, df in dfs_ts.items()]
    dfs_processed = pd.concat(dfs_processed).reset_index().drop(labels=["index"], axis=1)

    # Save dataframe to csv-file
    if append:
        dfs_processed.to_csv(path_processed, mode='a')
    else:
        dfs_processed.to_csv(path_processed)

    return dfs_processed


def _train_test_split_idx_country(df_country: pd.DataFrame, train_size: float) -> Tuple[int, int, int]:
    start = (~df_country["next_months_revenue"].isna()).idxmax()
    end = (~df_country["next_months_revenue"].isna())[::-1].idxmax()
    train_end = start + round(train_size * (end - start))

    return start, train_end, end


def train_test_split(df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_test_split_idx_country = df.groupby("country").apply(_train_test_split_idx_country, train_size=train_size)
    df_train = pd.concat([df.loc[start:end] for start, end, _ in train_test_split_idx_country])
    df_test = pd.concat([df.loc[start:end] for _, start, end in train_test_split_idx_country])

    return df_train, df_test


