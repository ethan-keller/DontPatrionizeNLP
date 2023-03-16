from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocessing import get_train_dev_test_df
from sklearn.model_selection import train_test_split
from split import read_data


def add_labels(ax, x, y, leftgap=0, topgap=0):
    for i in range(len(x)):
        ax.text(
            i - leftgap,
            y[i] + topgap,
            f"{np.round(y[i] * 100, 3)}%",
            ha="center",
        )


def plot_label_distribution(df, ax, title, labels=None):
    ticks, counts = np.unique(df["label"], return_counts=True)

    density = counts / counts.sum()

    if labels is None:
        labels = ticks
    label_ticks = np.arange(len(labels))
    ax.bar(labels, density, color=["r", "b"], alpha=0.5, width=0.3)
    add_labels(ax, label_ticks, density, topgap=0.01)

    ax.set_ylabel("Percentage of Labels in the Dataset")
    ax.set_title(title)


def plot_input_length_distribution(df, ax, title, density=False, **kwargs):
    input_lengths = df["text"].str.len()
    ax.hist(input_lengths, bins=50, **kwargs)
    ax.set_title(title)
    ax.set_ylabel("Number of Inputs" if not density else "Density")
    ax.set_xlabel("Length of Inputs")


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "data" / "official" / "dontpatronizeme_v1.4" / "dontpatronizeme_pcl.tsv"

    train_config_path = Path(__file__).parent.parent / "data" / "official" / "train_semeval_parids-labels.csv"

    dev_config_path = Path(__file__).parent.parent / "data" / "official" / "dev_semeval_parids-labels.csv"

    split = get_train_dev_test_df(data_path, train_config_path, dev_config_path, dev_rate=0.25, seed=42)
    train_df = split["train"]
    dev_df = split["dev"]
    test_df = split["test"]

    # Plot the label distribution in the train, dev and test set in separate figures
    fig, (train_ax, dev_ax, test_ax) = plt.subplots(1, 3, figsize=(18, 6))
    labels = ["Not Patronizing", "Patronizing"]
    plot_label_distribution(train_df, train_ax, "Label Distribution in the Train Set", labels)
    plot_label_distribution(dev_df, dev_ax, "Label Distribution in the Internal Dev Set", labels)
    plot_label_distribution(test_df, test_ax, "Label Distribution in the Official Dev Set", labels)

    plt.tight_layout()
    plt.show()

    # Plot the no of tokens distribution in the train, dev and test set in separate figures
    fig, (train_ax, dev_ax, test_ax) = plt.subplots(1, 3, figsize=(18, 6))
    plot_input_length_distribution(
        train_df,
        train_ax,
        "Input Length Distribution in the Train Set",
        density=True,
        color="firebrick",
        alpha=0.5,
    )
    plot_input_length_distribution(
        dev_df,
        dev_ax,
        "Input Length Distribution in the Dev Set",
        density=True,
        color="darkorange",
        alpha=0.5,
    )
    plot_input_length_distribution(
        test_df,
        test_ax,
        "Input Length Distribution in the Test Set",
        density=True,
        color="forestgreen",
        alpha=0.5,
    )

    plt.tight_layout()
    plt.show()

    # Plot raw label distributionin the train, dev and test set in separate figures
    df = read_data(data_path)
    train_config_df = pd.read_csv(train_config_path, sep=",")
    dev_config_df = pd.read_csv(dev_config_path, sep=",")
    train_df = train_config_df.drop("label", axis=1).merge(df, on="par_id", how="inner")
    test_df = dev_config_df.drop("label", axis=1).merge(df, on="par_id", how="inner")
    train_df, dev_df = train_test_split(train_df, test_size=0.25, random_state=42)

    fig, (train_ax, dev_ax, test_ax) = plt.subplots(1, 3, figsize=(18, 6))
    plot_label_distribution(train_df, train_ax, "Label Distribution in the Train Set")
    plot_label_distribution(dev_df, dev_ax, "Label Distribution in the Internal Dev Set")
    plot_label_distribution(test_df, test_ax, "Label Distribution in the Official Dev Set")

    plt.tight_layout()
    plt.show()
