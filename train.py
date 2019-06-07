# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.

import os
import math
import json
import logging
from tqdm import tqdm
from pprint import pformat
from itertools import chain
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from pytorch_pretrained_bert import (OpenAIAdam, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from utils import get_dataset

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS   = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS  = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path",                type=str,   default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache",               type=str,   default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint",            type=str,   default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates",              type=int,   default=2, help="Number of candidates for training")
    parser.add_argument("--max_history",                 type=int,   default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size",            type=int,   default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size",            type=int,   default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr",                          type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef",                     type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef",                     type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm",                    type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs",                    type=int,   default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations",    type=int,   default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16",       type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    return parser.parse_args()


def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    
    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    
    instance["input_ids"]      = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"]   = len(instance["input_ids"]) - 1
    instance["lm_labels"]      = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    
    return instance, sequence

# >>
class PersonaChatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, padding):
        self.dataset          = dataset
        self.num_candidates   = dataset['num_candidates']
        self.padding          = padding
    
    def __getitem__(self, idx):
        
        first = self.num_candidates * idx
        last  = self.num_candidates * (idx + 1)
        
        tmp = [None] * len(MODEL_INPUTS)
        for i, model_input in enumerate(MODEL_INPUTS):
            if model_input == 'mc_labels':
                tmp[i] = [self.num_candidates - 1]
            else:
                tmp[i] = self.dataset[model_input][first:last]
        
        return tmp
    
    def __len__(self):
        return len(self.dataset['input_ids']) // self.num_candidates
    
    def collate_fn(self, batch):
        batch_dict = dict(zip(MODEL_INPUTS, list(zip(*batch))))
        
        # Flatten
        batch_dict = {k:list(chain(*v)) for k,v in batch_dict.items()}
        
        # Pad
        max_l = max(len(x) for x in batch_dict['input_ids'])
        
        batch_dict['input_ids']      = [x + [self.padding] * (max_l - len(x)) for x in batch_dict['input_ids']]
        batch_dict['token_type_ids'] = [x + [self.padding] * (max_l - len(x)) for x in batch_dict['token_type_ids']]
        batch_dict['lm_labels']      = [x + [-1] * (max_l - len(x)) for x in batch_dict['lm_labels']]
        
        # Convert to tensor
        batch_dict = {k:torch.tensor(v) for k,v in batch_dict.items()}
        
        # Rehape
        for k,v in batch_dict.items():
            if k != 'mc_labels':
                shape = (-1, self.num_candidates) + v.shape[1:]
                v     = v.view(shape)
            
            batch_dict[k] = v
        
        return list(batch_dict.values())
# <<

def get_data_loaders(args, tokenizer):
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    
    # --
    # Build data
    
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        
        datasets[dataset_name]["num_candidates"] = num_candidates
        
        for dialog in tqdm(dataset):
            persona = dialog["personality"].copy()
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    
                    history    = utterance["history"][-(2 * args.max_history + 1):]
                    candidates = utterance["candidates"][-num_candidates:]
                    
                    for j, candidate in enumerate(candidates):
                        instance, _ = build_input_from_segments(persona, history, candidate, tokenizer, lm_labels=(j == num_candidates - 1))
                        
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                
                persona = [persona[-1]] + persona[:-1]  # permuted personalities
    
    # --
    # Make DataLoader
    
    logger.info("Build train and validation dataloaders")
    
    padding = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    
    train_dataset = PersonaChatDataset(datasets['train'], padding=padding)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_loader  = DataLoader(train_dataset, 
        collate_fn=train_dataset.collate_fn,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        shuffle=(not args.distributed),
    )
    
    valid_dataset = PersonaChatDataset(datasets['valid'], padding=padding)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    valid_loader  = DataLoader(valid_dataset,
        collate_fn=valid_dataset.collate_fn,
        sampler=valid_sampler,
        batch_size=args.valid_batch_size,
        shuffle=False,
    )
    
    return train_loader, valid_loader, train_sampler, valid_sampler


def train():
    
    # --
    # Initialization
    
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank) 
    logger.info("Arguments: %s", pformat(args))
    
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    # --
    # Define model
    
    logger.info("Prepare tokenizer, pretrained model and optimizer - add special tokens for fine-tuning")
    
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer
    model_class     = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model     = model_class.from_pretrained(args.model_checkpoint).to(args.device)
    
    tokenizer.set_special_tokens(SPECIAL_TOKENS)
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    
    optimizer = OpenAIAdam(model.parameters(), lr=args.lr)
    
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # --
    # Define datasets
    
    logger.info("Prepare datasets")
    
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)
    
    # --
    # Ops
    
    def update(engine, batch):
        _ = model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        lm_loss, mc_loss = model(*batch)
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return loss.item()
    
    VERBOSE = True
    def inference(engine, batch):
        _ = model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            
            if VERBOSE:
                logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            
            model_outputs = model(input_ids, mc_token_ids, token_type_ids=token_type_ids)
            
            lm_logits, mc_logits   = model_outputs[0], model_outputs[1]  # So we can also use GPT2 outputs
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    
    trainer   = Engine(update)
    evaluator = Engine(inference)
    
    # --
    # Callbacks
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))
    
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))
    
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    
    # --
    # Metrics
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    
    metrics = {
        "nll"      : Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
        "accuracy" : Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))
    }
    
    metrics.update({
        "average_nll"      : MetricsLambda(average_distributed_scalar, metrics["nll"], args),
        "average_accuracy" : MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)
    })
    
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    
    # --
    # Logging
    
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(json.dumps(evaluator.state.metrics)))
        
        # tb_logger = TensorboardLogger(log_dir=None)
        # tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        # tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        # tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)
        
        # checkpoint_handler = ModelCheckpoint(tb_logger.writer.log_dir, 'checkpoint', save_interval=1, n_saved=3)
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation
        
        # torch.save(args, tb_logger.writer.log_dir + '/model_training_args.bin')
        # getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.log_dir, CONFIG_NAME))
        # tokenizer.save_vocabulary(tb_logger.writer.log_dir)
        
    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)
    
    # # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    # if args.local_rank in [-1, 0] and args.n_epochs > 0:
    #     os.rename(checkpoint_handler._saved[-1][1][-1], os.path.join(tb_logger.writer.log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
    #     tb_logger.close()

if __name__ == "__main__":
    train()
