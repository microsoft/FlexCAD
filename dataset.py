import os
import random
import pickle
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer

IGNORE_INDEX = -100
MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class CADDataset(Dataset):
    def __init__(self, pickle_fn, llama_tokenizer=None):
        if not os.path.exists(pickle_fn):
            raise ValueError(f"{pickle_fn} does not exist")
        self.inputs = pickle.load(open(pickle_fn, "rb"))
        self.llama_tokenizer = llama_tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        if not 0 <= index < len(self.inputs):
            raise ValueError(f"Index {index} out of range")
        val = self.inputs[index]
        val = self.tokenize(val)
        return val

    def tokenize(self, input_str):
        if random.random() < 0.66:
            tokens = self.generation_task(input_str)
        else:
            tokens = self.infill_task(input_str)
        input_ids = labels = tokens.input_ids[0]
        input_id_lens = label_lens = (
            tokens.input_ids.ne(self.llama_tokenizer.pad_token_id).sum().item()
        )
        return dict(
            input_ids=input_ids,
            input_id_lens=input_id_lens,
            labels=labels,
            label_lens=label_lens,
        )

    def generation_task(self, input_str):
        tokens = self.llama_tokenizer(
            input_str,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        return tokens

    def infill_task(self, input_str):
        tokens = self.llama_tokenizer(
            input_str,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            truncation=True,
        )
        return tokens


if __name__ == "__main__":
    dataset = CADDataset(
        "val.pkl",
        llama_tokenizer=LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            model_max_length=MAX_LENGTH,
            padding_side="right",
            use_fast=False,
            pad_token=DEFAULT_PAD_TOKEN,
        ),
    )
    print(len(dataset))
    print(dataset[20])
