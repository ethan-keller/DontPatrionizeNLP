import argparse


def get_args(custom_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="./data/official/dontpatronizeme_v1.4/dontpatronizeme_pcl.tsv",
        help="The TSV file path to all raw text data of train and dev set.",
    )
    parser.add_argument(
        "--train-label-path",
        type=str,
        default="./data/official/train_semeval_parids-labels.csv",
        help="The CSV file path to the training labels. The official training set will be seperated into own train and dev set later.",
    )
    parser.add_argument(
        "--dev-label-path",
        type=str,
        default="./data/official/dev_semeval_parids-labels.csv",
        help="The CSV file path to the official dev labels. This will be used as test set in our case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed to everything for reproductability.",
    )
    parser.add_argument(
        "-nc",
        "--num-classes",
        type=int,
        default=2,
        help="The number of classes in our data. Normally is 2.",
    )
    parser.add_argument(
        "-mname",
        "--model-name",
        type=str,
        default="roberta-base",
        help="The pretrained model name of RoBERTa.",
    )
    parser.add_argument(
        "--dev-rate",
        type=float,
        default=0.25,
        help="The portion of splitted dev set in the whole official train set.",
    )
    parser.add_argument(
        "-mlen",
        "--max-text-length",
        type=int,
        default=256,
        help="The maximum length (number of tokens) in model. All sentences will be padded to that length.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="The batch size.")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="The name of the device. Use cpu if no GPU available.",
    )
    # Deprecated training arguments.
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=3e-5,
        help="The learning rate.",
    )
    # Deprecated training arguments.
    parser.add_argument(
        "-wd",
        "--weight-decay",
        type=float,
        default=1e-5,
        help="The weight decay for L2 regularization.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="The patience for early stopping monitored by dev set F1 score.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints",
        help="The path to save best model.",
    )
    # Random search paramter.
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="The number of iterations for parameter optimizing.",
    )
    # Random search parameter.
    parser.add_argument(
        "--n-param-iter",
        type=int,
        default=2,
        help="The number of iterations for each set of parameters.",
    )
    parser.add_argument(
        "-ln",
        "--loss-name",
        type=str,
        default="cross_entropy",
        choices=[
            "cross_entropy",
            "weighted_cross_entropy",
            "focal",
            "label_smoothing",
            "bce",
        ],
        help="The loss type.",
    )
    parser.add_argument(
        "-lp", "--log-path", type=str, default="param-log.csv", help="The path to the log of tried param results."
    )
    parser.add_argument(
        "-aug", "--augmentation", action="store_true", help="Whether use data augmentation (random word swapping)."
    )
    parser.add_argument(
        "-cbs", "--class-balanced-sampling", action="store_true", help="Whether use class balanced sampling."
    )

    # Prediction parameter.
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        default=None,
        help="The trained model path. Only used in predict.py.",
    )
    parser.add_argument(
        "--held-out-data-path",
        type=str,
        default="./data/official/task4_test.tsv",
        help="The path to held-out test data tsv.",
    )
    parser.add_argument(
        "--test-pred-path",
        type=str,
        default="./dev.txt",
        help="The output text file of predctions of test set.",
    )
    parser.add_argument(
        "--held-out-pred-path",
        type=str,
        default="./test.txt",
        help="The output text file of predctions of held-out test set.",
    )

    # Ignore unknown arguments (needed if run with ipykernel)
    args, _ = parser.parse_known_args(custom_args)
    return args
