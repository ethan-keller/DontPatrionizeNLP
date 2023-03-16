import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from utils.split import read_data, split_data
import nlpaug.augmenter.word as naw


def get_train_dev_test_df(raw_data_path, train_label_path, dev_label_path, dev_rate, seed):
    df = read_data(raw_data_path)
    train_df, test_df = split_data(df, train_label_path, dev_label_path)
    train_df, dev_df = train_test_split(train_df, test_size=dev_rate, random_state=seed)
    return {"train": train_df, "dev": dev_df, "test": test_df}


def get_prediction_df(raw_data_path):
    return read_data(raw_data_path, test_mode=True)


def encode_text(text, tokenizer, max_text_length):
    """A helper function to encode the text into input ids, attention masks and labels."""
    res = tokenizer.encode_plus(
        text,
        padding="max_length",
        max_length=max_text_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return res


def df_to_dataset(df, tokenizer, max_text_length, test_mode=False, augmentation=True):
    aug = naw.RandomWordAug(action="swap")
    if augmentation:
        print("Augmenting data...")
    encode_results = (
        df["text"]
        .fillna("")
        .apply(lambda x: encode_text(aug.augment(x)[0] if (augmentation and x) else x, tokenizer, max_text_length))
    )
    if augmentation:
        print("Done augmenting data")
    # split the results into input ids and attention masks
    all_ids = torch.tensor([x["input_ids"] for x in encode_results])
    all_masks = torch.tensor([x["attention_mask"] for x in encode_results])

    if not test_mode:
        # The labels in the dataframe are int64, but the model expects longs.
        all_labels = torch.LongTensor(df["label"].values)
        dataset = TensorDataset(all_ids, all_masks, all_labels)
    else:
        dataset = TensorDataset(all_ids, all_masks)
    return dataset
