# 对模型进行MLM预训练
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import math
from transformers import BertTokenizer, LongformerModel, LongformerTokenizer, BertModel, BertConfig, BertForMaskedLM
from queue import Queue
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertTokenizerFast, ErnieModel, ErnieForMaskedLM
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import random
from collections import Counter
import pickle
import numpy as np
import argparse
import torch.nn.functional as F
import os

# import warnings
# warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=32,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='./pretrained-models/', 
        help="Where to store the final model."
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    args = parser.parse_args()
    return args


# 定义一个数据集
class MyDataset(Dataset):
    def __init__(self, inputs):
        self.examples = []
        for input_ids, token_type_ids in zip(inputs["input_ids"], inputs["attention_mask"]):
            self.examples.append(
                {"input_ids": input_ids, "attention_mask": token_type_ids})
        self.length = len(inputs['input_ids'])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.examples[index]


def main(args):
    # 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    print("loading add_pretrain_model_path=", args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    if args.resume_from_checkpoint:
        tokenizer = AutoTokenizer.from_pretrained(args.resume_from_checkpoint, use_fast=True)
        model = AutoModelForMaskedLM.from_pretrained(args.resume_from_checkpoint)
        print("loading checkpoint to continue training!")

    sentences = pd.read_csv(args.train_file)['text'].tolist()
    token_feat = tokenizer.batch_encode_plus(
        sentences,
        max_length=args.max_seq_length,
        return_tensors='pt',
        padding='max_length',
        truncation=True
    )

    train_dataset = MyDataset(token_feat)

    print(train_dataset[0])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)



if __name__ == '__main__':
    args = parse_args()
    main(args)

'''
#创建一个pretrain_models的文件夹
python pretrain_2.py

'''
