# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import logging
import tarfile
import tempfile
from tqdm import tqdm
from joblib import Parallel, delayed

import torch

from pytorch_pretrained_bert import cached_path

PERSONACHAT_URL    = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

logger = logging.getLogger(__file__)

# --
# Helpers

def tokenize(tokenizer, obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(tokenizer, o)) for n, o in obj.items())
    
    if len(obj) > 100:
        jobs = [delayed(tokenize)(tokenizer, o) for o in obj]
        return Parallel(backend='multiprocessing', n_jobs=-1, verbose=10)(jobs)
    else:
        return list(tokenize(tokenizer, o) for o in obj)


def download_pretrained_model():
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    
    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    
    return tempdir

# --
# Dataset utilities

def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    dataset_path  = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        
        logger.info("Tokenize and encode the dataset")
        dataset = tokenize(tokenizer, dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    
    return dataset


def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    dataset_path  = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())
            
        logger.info("Tokenize and encode the dataset")
        personachat = tokenize(tokenizer, personachat)
        torch.save(personachat, dataset_cache)
    
    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])
    
    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

