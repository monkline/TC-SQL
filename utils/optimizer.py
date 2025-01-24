'''
Author: whateverisnottaken johnzhaozitian@gmail.com
Date: 2024-01-07 11:50:50
LastEditors: whateverisnottaken johnzhaozitian@gmail.com
LastEditTime: 2024-01-16 09:10:08
FilePath: /ssd/agents/RAG_demo/RSTC/utils/optimizer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch.optim.lr_scheduler import StepLR
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
import os
import torch
BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased',
    "bert": 'bert-base-uncased',
    "roberta": 'roberta-base',
}

SBERT_CLASS = {
    "distilbert": './all-MiniLM-L6-v2',
    "MiniLM": './distilbert-base-nli-stsb-mean-tokens',
    "bge-base-en-v1.5": '/home/calf/ssd/models/bge-base-en-v1.5',
    'AG': 'pretrained-models/AG/12000',  # 12000   0.8521
    'SO': 'pretrained-models/SO/26000',  # 30000   88.3-79.48
    'SS': 'pretrained-models/SS/12000',
    'Bio': 'pretrained-models/Bio/10000',  # 9000
    'G-TS': 'pretrained-models/G-TS/20000',  #20000    0.8871  0.9624
    'G-T': 'pretrained-models/G-T/5000',
    'G-S': 'pretrained-models/G-S/20000',
    'Tweet': 'pretrained-models/Tweet/2000',
}


def get_optimizer(model, args):
    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters()},
        {'params': model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params': model.adapt.parameters(), 'lr': args.lr * args.lr_scale},
        {'params': model.prob.parameters(), 'lr': args.lr * args.lr_scale},
    ], lr=args.lr)

    return optimizer


def get_bert(args):

    if args.use_pretrain == "SBERT":
        # bert_model = get_sbert(args)
        # tokenizer = bert_model[0].tokenizer
        # model = bert_model[0].auto_model
        config = AutoConfig.from_pretrained('pretrained-models/distilbert-base-nli-stsb-mean-tokens')
        model = AutoModel.from_pretrained(SBERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained('pretrained-models/distilbert-base-nli-stsb-mean-tokens')
        print("..... loading Sentence-BERT !!!")
    else:
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")

    return model, tokenizer


def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert


'''
zip -r all.zip ./ -x '/home/calf/ssd/agents/RAG_demo/RSTC/pretrained-models/*' -x '/home/calf/ssd/agents/RAG_demo/TextClu/*'
'''