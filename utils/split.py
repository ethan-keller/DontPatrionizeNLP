from pathlib import Path

import pandas as pd


def read_data(data_path, test_mode=False):
    """Reads the PCL data from the given path and returns a pandas dataframe"""
    names = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    skiprows = [0, 1, 2]
    if test_mode:
        names = names[:-1]
        skiprows = []
    df = pd.read_csv(
        data_path,
        sep="\t",
        skiprows=skiprows,
        names=names,
    )
    return df


def split_data(
    df,
    train_config_path,
    dev_config_path,
):
    """Splits the data into train and dev sets based on the given configuration files"""
    train_config_df = pd.read_csv(train_config_path, sep=",")
    dev_config_df = pd.read_csv(dev_config_path, sep=",")
    train_df = train_config_df.drop("label", axis=1).merge(
        df, on="par_id", how="inner"
    )
    train_df.loc[train_df["label"] <= 1, "label"] = 0
    train_df.loc[train_df["label"] > 1, "label"] = 1
    dev_df = dev_config_df.drop("label", axis=1).merge(
        df, on="par_id", how="inner"
    )
    dev_df.loc[dev_df["label"] <= 1, "label"] = 0
    dev_df.loc[dev_df["label"] > 1, "label"] = 1
    return train_df, dev_df


if __name__ == "__main__":
    # Replace this path with the path to the data file
    data_path = (
        Path(__file__).parent.parent
        / "data"
        / "official"
        / "dontpatronizeme_v1.4"
        / "dontpatronizeme_pcl.tsv"
    )
    train_config_path = (
        Path(__file__).parent.parent
        / "data"
        / "official"
        / "train_semeval_parids-labels.csv"
    )

    dev_config_path = (
        Path(__file__).parent.parent
        / "data"
        / "official"
        / "dev_semeval_parids-labels.csv"
    )

    df = read_data(data_path)
    train_df, dev_df = split_data(df, train_config_path, dev_config_path)
    print(train_df)
