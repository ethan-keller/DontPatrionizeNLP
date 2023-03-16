import csv
import os
from copy import deepcopy

import numpy as np
import torch
from arguments import get_args
from scipy.stats import uniform, norm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers.utils import logging
from utils.losses import get_loss_fun
from utils.preprocessing import (
    df_to_dataset,
    get_prediction_df,
    get_train_dev_test_df,
)

logging.set_verbosity(40)


def train_dev_model(
    model,
    df_partitions,
    tokenizer,
    epochs,
    batch_size,
    loss_fun,
    patience,
    class_balanced_sampling,
    augmentation,
    device,
    max_text_length,
    learning_rate,
    weight_decay,
    beta1,
):
    # Convert the dataframes into datasets with input ids, attention masks and labels.
    datasets = {
        part: df_to_dataset(df, tokenizer, max_text_length, augmentation = augmentation and part == "train")
        for part, df in df_partitions.items()
    }

    # Balance classes when sampling by weighing each class by inverted frequency
    dataloaders = {}
    for part, dataset in datasets.items():
        if class_balanced_sampling and part == "train":
            targets = dataset.tensors[2]
            counts = np.unique(targets, return_counts=True)[1]
            class_weights = [np.sum(counts) / c for c in counts]
            print(f"Class weights: {class_weights}")
            weights = [class_weights[c] for c in targets]
            dataloaders[part] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle = part == "train",
                sampler=WeightedRandomSampler(weights, len(targets)),
            )
        else:
            dataloaders[part] = DataLoader(dataset, batch_size = batch_size, shuffle = part == "train")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, 0.999),
    )
    logs = {"train": {"F1": [], "loss": []}, "dev": {"F1": [], "loss": []}}
    max_f1, stop_training = -1e10, False
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        for phase in ["train", "dev"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            losses, y_trues, y_preds = [], [], []
            for input_ids, attention_mask, labels in dataloaders[phase]:
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.view(-1).to(device),
                )
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(input_ids, attention_mask, labels=None)[0]
                    loss = loss_fun(logits, labels)
                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                y_trues.append(labels.cpu().numpy())
                y_preds.append(logits.max(dim=-1)[1].detach().cpu().numpy())
                losses.append(loss.item())
            loss, y_trues, y_preds = (
                sum(losses) / len(losses),
                np.concatenate(y_trues),
                np.concatenate(y_preds),
            )
            this_f1 = f1_score(y_trues, y_preds)
            print(
                f"{phase.capitalize()}: Loss: {loss:.4f}, F1 score: {this_f1:.4f}"
            )
            logs[phase]["loss"].append(loss)
            logs[phase]["F1"].append(this_f1)
            if phase == "dev":
                if this_f1 > max_f1:
                    max_f1, max_epoch = this_f1, epoch
                    best_state_dict = deepcopy(model.state_dict())
                elif epoch - max_epoch >= patience:
                    stop_training = True
        if stop_training:
            break
    return max_f1, logs, best_state_dict


def write_record(path, params, max_f1):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "max_text_length",
                "learning_rate",
                "weight_decay",
                "beta1",
                "f1",
            ],
        )
        writer.writerow({**params, "f1": max_f1})


def main():
    # Getting the default arguments for training.
    # The arguments are:
    #   1. Paths to data and labels, including the raw data path,
    # 	  the official train/dev label path
    #   2. The seed for reproductability
    #   3. Model configurations parameters, including the pretrained model name,
    # 	  the number of classes, the maximum length of text
    #   4. Training parameters, including the internal train-dev(i.e. validation)
    # 	  split rate, and the hyperparameters
    # 	  e.g. the batch size, the number of epochs, the learning rate etc.
    #   5. The device to use, e.g. CPU or GPU and the path to store the model

    args = get_args()
    device = args.device
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    # Split the data into train, dev and test dataframes and store them in a dictionary
    df_partitions = get_train_dev_test_df(
        args.raw_data_path,
        args.train_label_path,
        args.dev_label_path,
        args.dev_rate,
        args.seed,
    )
    loss_fun = get_loss_fun(args.loss_name)
    # Round 1 Params
    """
    param_distribution = {
        "max_text_length": [16, 32, 64, 128, 256, 512],
        "learning_rate": uniform(loc=1e-6, scale=1e-4),
        "weight_decay": uniform(loc=0, scale=1e-4),
        "beta1": uniform(loc=0.1, scale=0.8),
    }
    """
    # Round 2 Params
    """
	param_distribution = {
		"max_text_length": [64, 128, 256],
		"learning_rate": uniform(loc = 1e-5, scale = 3e-5),
		"weight_decay": uniform(loc = 2e-6, scale = 7e-5),
		"beta1": uniform(loc = 0.55, scale = 0.35),
	}
	"""
    # Round 3 Params
    param_distribution = {
        "max_text_length": [256],
        "learning_rate": norm(loc = 2e-5, scale = 1e-5),
        "weight_decay": norm(loc = 4e-5, scale = 2e-5),
        "beta1": norm(loc = 0.8, scale = 0.1),
    }

    overall_max_f1 = -1e10
    for params in ParameterSampler(param_distribution, n_iter=args.n_iter):
        for __ in range(args.n_param_iter):
            print(params)
            model = RobertaForSequenceClassification.from_pretrained(
                args.model_name, num_labels=args.num_classes
            ).to(device)
            max_f1, logs, best_state_dict = train_dev_model(
                model,
                df_partitions,
                tokenizer,
                args.epochs,
                args.batch_size,
                loss_fun,
                args.patience,
                args.class_balanced_sampling,
                args.augmentation,
                device,
                **params,
            )
            write_record(
                os.path.join(args.save_path, "param-log.csv"), params, max_f1
            )
            if overall_max_f1 < max_f1:
                print("Found new best F1: ", max_f1)
                try:
                    os.remove(os.path.join(args.save_path, "model-optim-{:.4f}.pt".format(overall_max_f1)))
                except Exception as e:
                    print(e)
                overall_max_f1, optim_param = max_f1, deepcopy(params)
                overall_best_state_dict = best_state_dict
                if max_f1 > 0.6:
                    torch.save(
                        best_state_dict,
                        os.path.join(
                            args.save_path,
                            "model-optim-{:.4f}.pt".format(max_f1),
                        ),
                    )
                    logs = np.stack(
                        [
                            logs["train"]["F1"],
                            logs["train"]["loss"],
                            logs["dev"]["F1"],
                            logs["dev"]["loss"],
                        ]
                    )
                    np.save(
                        os.path.join(
                            args.save_path,
                            "train-log-{:.4f}.npy".format(max_f1),
                        ),
                        logs,
                    )

    print("Optimal params: ", optim_param)


if __name__ == "__main__":
    main()
