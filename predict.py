import numpy as np
import torch
from arguments import get_args
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers.utils import logging
from utils.preprocessing import (
    df_to_dataset,
    get_prediction_df,
    get_train_dev_test_df,
)

logging.set_verbosity(40)


def main():
    args = get_args()
    device = args.device
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    df_partitions = get_train_dev_test_df(
        args.raw_data_path,
        args.train_label_path,
        args.dev_label_path,
        args.dev_rate,
        args.seed,
    )
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_dataset = df_to_dataset(df_partitions["test"], tokenizer, args.max_text_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    y_trues, y_preds = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_dataloader:
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            loss, logits = model(input_ids, attention_mask, labels=labels)[:2]
            y_trues.append(labels.cpu().numpy())
            y_preds.append(logits.max(dim=-1)[1].cpu().numpy())
    y_trues = np.concatenate(y_trues).flatten()
    y_preds = np.concatenate(y_preds)
    np.save("yt.npy", y_trues)
    np.save("yp.npy", y_preds)
    print(y_trues, y_preds)
    test_f1 = f1_score(y_trues, y_preds, pos_label=1, average="binary")
    np.savetxt(args.test_pred_path, y_preds.astype(np.int64), fmt="%i", delimiter="\n")
    np.savetxt("dev_yt.txt", y_trues.astype(np.int64), fmt="%i", delimiter="\n")
    print("Model: {} Test F1: {:.4f}".format(args.model_path, test_f1))

    # exit()

    ho_df = get_prediction_df(args.held_out_data_path)
    ho_dataset = df_to_dataset(ho_df, tokenizer, args.max_text_length, test_mode=True)
    ho_dataloader = DataLoader(ho_dataset, batch_size=args.batch_size, shuffle=False)
    y_preds = []
    with torch.no_grad():
        for input_ids, attention_mask in ho_dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            logits = model(input_ids, attention_mask, labels=None)[0]
            y_preds.append(logits.max(dim=-1)[1].cpu().numpy())
    y_preds = np.concatenate(y_preds)
    np.savetxt(
        args.held_out_pred_path,
        y_preds.astype(np.int64),
        fmt="%i",
        delimiter="\n",
    )


if __name__ == "__main__":
    main()
