import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from arguments import get_args
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from tqdm import trange
from utils.split import read_data, split_data


class BoWNNClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(BoWNNClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, x):
        return self.linear(x)


def construct_vocab(X):
    """Construct a vocabulary mapping word to index for a given corpus."""
    word_to_idx = {}
    for sentence in X:
        for word in preprocess_text(sentence):
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def preprocess_text(text):
    """Preprocess the text data."""
    # tokenization
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)

    # remove stop words
    tokens = [w for w in tokens if w not in stop_words]

    # remove punctuation
    tokens = [w for w in tokens if w.isalpha()]

    # stemming
    tokens = list(set([stemmer.stem(w) for w in tokens]))

    return tokens


def construct_bow_vector(sentence, word_to_idx):
    """Convert a sentence into a bag-of-words vector."""
    vec = torch.zeros(len(word_to_idx))
    for word in preprocess_text(sentence):
        vec[word_to_idx[word]] += 1
    return vec.view(1, -1)


def construct_bow_matrix(X, word_to_idx):
    """Convert a corpus into a bag-of-words matrix."""
    return torch.cat([construct_bow_vector(sentence, word_to_idx) for sentence in X])


if __name__ == "__main__":
    # Getting the default arguments for training.
    baseline_parser = argparse.ArgumentParser()
    baseline_parser.add_argument(
        "--classifier",
        type=str,
        default="nn",
        help="The model to use for training. Options are 'nn', 'logistic', 'svm' and 'nb'.",
    )

    baseline_args = baseline_parser.parse_args()
    model_name = baseline_args.classifier
    if model_name not in ["nn", "logistic", "svm", "nb", "rf", "xgb"]:
        raise ValueError("The model must be one of 'nn', 'logistic', 'svm', 'nb', 'rf' or 'xgb'.")

    args = get_args()
    train_label_path = args.train_label_path
    dev_label_path = args.dev_label_path
    raw_data_path = args.raw_data_path
    batch_size = args.batch_size
    df = read_data(raw_data_path)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Split the data into train and dev dataframes
    train_df, dev_df = split_data(df, train_label_path, dev_label_path)

    # Construct the vocabulary and the bag-of-words matrix
    X_train = train_df["text"].values.astype("U")
    X_dev = dev_df["text"].values.astype("U")

    word_to_idx = construct_vocab(np.concatenate((X_train, X_dev)))
    vocab_size = len(word_to_idx)
    num_classes = 2

    X_train_bow = construct_bow_matrix(X_train, word_to_idx)
    y_train = train_df["label"].values
    X_dev_bow = construct_bow_matrix(X_dev, word_to_idx)
    y_dev = dev_df["label"].values

    # get the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Construct a dataloader for the training set
    train_dataset = torch.utils.data.TensorDataset(X_train_bow, torch.tensor(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if model_name == "nn":
        model = BoWNNClassifier(vocab_size, num_classes).to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Training loop
        for epoch in trange(args.epochs):
            for bow_mat, labels in train_loader:
                bow_mat = bow_mat.to(device)
                labels = labels.to(device)
                output = model(bow_mat)
                loss = loss_function(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(
            model.state_dict(),
            Path(__file__).parent / "checkpoints" / "bow-nn.pt",
        )
        # Constrect a dataloader for the dev set
        dev_dataset = torch.utils.data.TensorDataset(X_dev_bow, torch.tensor(y_dev))
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=1, shuffle=False)

        # Evaluation on the dev set
        with torch.no_grad():
            y_preds = []
            for bow_mat, labels in dev_loader:
                bow_mat = bow_mat.to(device)
                output = model(bow_mat)
                y_preds.append(output.argmax().item())
    else:
        if model_name == "logistic":
            model = LogisticRegression(C=3)
            model.fit(X_train_bow, y_train)
            torch.save(
                model,
                Path(__file__).parent / "checkpoints" / "bow-logistic.pt",
            )
        elif model_name == "svm":
            model = LinearSVC(C=3)
            model.fit(X_train_bow, y_train)
            torch.save(model, Path(__file__).parent / "checkpoints" / "bow-svm.pt")
        elif model_name == "nb":
            model = MultinomialNB()
            model.fit(X_train_bow, y_train)
            torch.save(model, Path(__file__).parent / "checkpoints" / "bow-nb.pt")
        elif model_name == "rf":
            model = RandomForestClassifier()
            model.fit(X_train_bow, y_train)
            torch.save(model, Path(__file__).parent / "checkpoints" / "bow-rf.pt")
        elif model_name == "xgb":
            model = GradientBoostingClassifier()
            model.fit(X_train_bow, y_train)
            torch.save(model, Path(__file__).parent / "checkpoints" / "bow-xgb.pt")
        y_preds = model.predict(X_dev_bow)

    print(f"F1 score on dev set: {f1_score(y_dev, y_preds):.4f} with model {model.__class__.__name__}")
