# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import sys
import argparse
import glob
import logging
import os
import random
import timeit
import json
import copy
import math
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data import Subset, ConcatDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from pathlib import Path
import spacy
from pytorch_memlab import profile

from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadExample, SquadFeatures

from squad import SquadResult, SquadV1Processor, SquadV2Processor
from utils import Statistics, load_json, save_json
from utils_qa import get_bool_of_biased_dataset

import warnings
warnings.simplefilter('ignore')

import wandb

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
}
from transformers import __version__

MODEL_TYPES = list(MODEL_CLASSES.keys())


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, train_dataset, train_dataloader, model, tokenizer, wandb, optimizer, scheduler, t_total, max_steps=0, num_epochs=0, global_step=0, logging_steps=-1):
    """ Train the model """

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    if args.local_rank == -1 and args.evaluate_during_training and args.log_before_train:
        results = evaluate(args, model, tokenizer, prefix=global_step)
        metrics = {}
        for key, value in results.items():
            metrics[f"eval/{key}"] = value
        wandb.log(metrics, step=global_step)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type.split('_')[0] in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type.split('_')[0] in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs, return_dict=True)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0]:
                    metrics = {}
                    metrics["train/lr"] = scheduler.get_lr()[0]
                    metrics["train/loss"] = loss

                    if logging_steps > 0 and args.evaluate_during_training and global_step % logging_steps == 0:
                        results = evaluate(args, model, tokenizer, prefix=global_step)
                        for key, value in results.items():
                            metrics[f"eval/{key}"] = value

                    wandb.log(metrics, step=global_step)

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    return global_step, loss


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, 'dev', args.predict_file, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            if args.model_type == 'bert_bias_rel_pos':
                inputs = convert_batch_to_relative_position_inputs(args, tokenizer, batch)
            else:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

            if args.model_type.split('_')[0] in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type.split('_')[0] in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs, return_dict=True)

        for i, feature_index in enumerate(feature_indices):
            # TODO: i and feature_index are the same number! Simplify by removing enumerate?
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if args.model_type.split('_')[0] in ["xlnet", "xlm"] and args.model_type is not 'xlnet_simple':
                start_logits = outputs.start_top_log_probs[i]
                start_top_index = outputs.start_top_index[i]
                end_logits = outputs.end_top_log_probs[i]
                end_top_index = outputs.end_top_index[i]
                cls_logits = outputs.cls_logits[i]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits = to_list(outputs.start_logits[i])
                end_logits = to_list(outputs.end_logits[i])
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    pred_file_id = Path(args.predict_file).stem
    output_prediction_file = os.path.join(args.output_dir, f"predictions_{pred_file_id}_{prefix}.json")
    output_nbest_file = os.path.join(args.output_dir, f"nbest_predictions_{pred_file_id}_{prefix}.json")
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, f"null_odds_{pred_file_id}_{prefix}.json")
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type.split('_')[0] in ["xlnet", "xlm"] and args.model_type is not 'xlnet_simple':
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(args, tokenizer, mode, file, output_examples=False):
    if args.local_rank not in [-1, 0] and not mode in ['dev', 'test']:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else os.path.dirname(file)
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            os.path.splitext(os.path.basename(file))[0],
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        #if (not args.data_dir and\
        #    ((mode in ['dev', 'test'] and not args.predict_file) or\
        #    (mode in ['train'] and not args.train_file) or\
        #    (mode in ['pretrain'] and not args.pretrain_file))):
        if False:
            # We do not want to use the original split of squad.
            assert False
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=mode in ['dev', 'test'])
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if mode in ['dev', 'test']:
                examples = processor.get_dev_examples(args.data_dir, filename=file, do_lower_case='triviaqa' in input_dir)
            elif mode == 'train':
                examples = processor.get_train_examples(args.data_dir, filename=file, do_lower_case='triviaqa' in input_dir)
            elif mode == 'pretrain':
                examples = processor.get_train_examples(args.data_dir, filename=file, do_lower_case='triviaqa' in input_dir)
            # do_lower_case here for processing triviaqa

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=mode in ['train', 'pretrain'],
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and mode in ['train', 'pretrain']:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def prepare_optimizer_and_schedule(args, model, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_ratio > 0:
        args.warmup_steps = int(t_total * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    return optimizer, scheduler


def mdl_probe(args, model, tokenizer, prefix=""):
    """
    model: transformer (bert, roberta, ...)
    """
    for task_name in args.probe_tasks:
        reporting_root = os.path.join(args.output_dir, 'probing_'+task_name+'_L' + str(args.probe_layer) + '_'+str(prefix))
        if not os.path.exists(reporting_root):
            os.makedirs(reporting_root)
        args.reporting_root = reporting_root

        dataset_class = choose_dataset_class(args.model_type.split('_')[0])
        task_class, reporter_class, loss_class = choose_task_classes(task_name)
        probe_class = choose_probe_class(args, task_name)
        regimen_class = regimen.ProbeRegimen

        task = task_class(args)
        expt_dataset = dataset_class(args, task, model, tokenizer)
        expt_reporter = reporter_class(args, expt_dataset, task_name)
        expt_probe = probe_class(args)
        expt_regimen = regimen_class(args, reporter=expt_reporter, task_name=task_name)
        expt_loss = loss_class(args)

        online_coding_list = []
        dev_report_list = []
        test_report_list = []

        dev_dataloader = expt_dataset.get_dev_dataloader()
        test_dataloader = expt_dataset.get_test_dataloader()

        shuffled_dataset = torch.utils.data.Subset(
            expt_dataset.train_dataset, indices=np.random.permutation(len(expt_dataset.train_dataset)))
        train_portions, eval_portions = split_data_into_portions(shuffled_dataset)
        for i in range(len(train_portions) - 1):
            expt_probe = probe_class(args)
            current_train = DataLoader(train_portions[i],
                                 batch_size=expt_dataset.batch_size,
                                 collate_fn=expt_dataset.custom_pad, shuffle=False)
            current_dev = DataLoader(eval_portions[i],
                                 batch_size=expt_dataset.batch_size,
                                 collate_fn=expt_dataset.custom_pad, shuffle=False)

            # run-train-probe
            reports, evals = expt_regimen.train_until_convergence(expt_probe, expt_loss,
                                                      current_train,
                                                      dev_dataloader,
                                                      eval_datasets = {'dev': dev_dataloader,
                                                                       'test': test_dataloader,
                                                                       'online_portion': current_dev})
            online_coding_list.append(evals)
            expt_probe.load_state_dict(torch.load(expt_regimen.params_path))
            expt_probe.eval()

            # eval on portion, dev/test
            dev_predictions = expt_regimen.predict(expt_probe, dev_dataloader)
            dev_report = expt_reporter(dev_predictions, dev_dataloader, 'dev', probe=expt_probe)
            dev_report_list.append(dev_report)

            test_predictions = expt_regimen.predict(expt_probe, test_dataloader)
            test_report = expt_reporter(test_predictions, test_dataloader, 'test', probe=expt_probe)
            print('\n\nTest Report: ', test_report, len(test_dataloader), '\n\n')
            test_report_list.append(test_report)
        expt_probe = probe_class(args)
        # train on the last portion
        current_train = DataLoader(train_portions[-1],
                                 batch_size=expt_dataset.batch_size,
                                 collate_fn=expt_dataset.custom_pad, shuffle=False)

        # run-train-probe
        reports, evals = expt_regimen.train_until_convergence(expt_probe, expt_loss,
                                                      current_train,
                                                      dev_dataloader,
                                                      eval_datasets = {'dev': dev_dataloader,
                                                                       'test': test_dataloader,
                                                                       'train': current_train,})
        online_coding_list.append(evals)

        # load best model from current iteration
        expt_probe.load_state_dict(torch.load(expt_regimen.params_path))
        expt_probe.eval()

        # eval on portion
        dev_predictions = expt_regimen.predict(expt_probe, dev_dataloader)
        dev_report = expt_reporter(dev_predictions, dev_dataloader, 'dev', probe=expt_probe)
        dev_report_list.append(dev_report)

        test_predictions = expt_regimen.predict(expt_probe, test_dataloader)
        test_report = expt_reporter(test_predictions, test_dataloader, 'test', probe=expt_probe)
        print('\n\nTest Report: ', test_report, len(test_dataloader), '\n\n')
        test_report_list.append(test_report)

    with open(os.path.join(args.reporting_root, 'online_coding.pkl'), 'wb') as f:
        pickle.dump(online_coding_list, f)
    with open(os.path.join(args.reporting_root, 'online_dev_report.json'), 'w') as f:
        json.dump(dev_report_list, f, indent=4)
    with open(os.path.join(args.reporting_root, 'online_test_report.json'), 'w') as f:
        json.dump(test_report_list, f, indent=4)
    return test_report_list[-1]


def evaluate_loss(args, dataloader, model):
    all_loss = 0
    num_examples = 0
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            if args.model_type.split('_')[0] in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]
            outputs = model(**inputs)
            loss = outputs.loss
            batch_size = batch[0].size(0)
            all_loss += loss.item() * batch_size
            num_examples += batch_size
    return all_loss / num_examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--output_dir_base",
        default="/data/shinoda/git/QuestionGeneration/output/",
        type=str,
        help="The output directory base where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default="/data/shinoda/dataset/qa/squad/train-v1.1.json",
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file_synonym",
        default="input/squad-du-split/train.tokenized-v1.0.synonym.json",
        type=str,
        help="The input training file.",
    )
    parser.add_argument(
        "--train_file_backed",
        default="input/squad-du-split/train.tokenized-v1.0.backed.json",
        type=str,
        help="The input training file.",
    )
    parser.add_argument(
        "--mix_train_file",
        default="",
        type=str,
        help="Mix train file used when do_mix_train",
    )
    parser.add_argument(
        "--pretrain_files",
        default=[],
        nargs='*',
        type=str
    )
    parser.add_argument(
        "--predict_file",
        default="/data/shinoda/dataset/qa/squad/dev-v1.1.json",
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--preprocess_file",
        default=None,
        type=str,
        help="The input file you want to preprocess.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_only_preprocess", action="store_true", help="Only preprocessing")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--do_online_code", action="store_true", help="Do online code analysis"
    )
    parser.add_argument(
        "--do_biased_train", action="store_true", help="Do biased train"
    )
    parser.add_argument(
        "--bias_1",
        default=None,
        type=str,
        help="Specify bias type for biased training"
    )
    parser.add_argument(
        "--bias_1_larger_than",
        type=float,
        default=None,
        help="examples where biases are greater than or equal to the threshold are used to train a model",
    )
    parser.add_argument(
        "--bias_1_smaller_than",
        type=float,
        default=None,
        help="examples where biases are smaller than or equal to the threshold are used to train a model",
    )
    parser.add_argument(
        "--bias_1_included_in",
        default=None,
        nargs='*',
        type=str,
        help="examples where biases are included in this list are used to train a model"
    )
    parser.add_argument(
        "--bias_1_custom_func",
        default=None,
        type=str,
        help="examples where the output of this func given a bias is true are used to train a model"
    )
    parser.add_argument(
        "--bias_1_same_as",
        default=None,
        type=str,
        help="examples where biases are same as other biases"
    )
    parser.add_argument(
        "--bias_1_not_equal",
        default=None,
        type=str,
        help="examples where biases are not equal to other biases"
    )
    parser.add_argument(
        "--bias_1_top_k",
        type=int,
        default=None,
        help="examples where biases are top k",
    )
    parser.add_argument(
        "--bias_2",
        default=None,
        type=str,
        help="Specify bias type for biased training"
    )
    parser.add_argument(
        "--bias_2_larger_than",
        type=float,
        default=None,
        help="examples where biases are greater than the threshold are used to train a model",
    )
    parser.add_argument(
        "--bias_2_smaller_than",
        type=float,
        default=None,
        help="examples where biases are smaller than the threshold are used to train a model",
    )
    parser.add_argument(
        "--bias_2_included_in",
        default=None,
        nargs='*',
        type=str,
        help="examples where biases are included in this list are used to train a model"
    )
    parser.add_argument(
        "--bias_2_custom_func",
        default=None,
        type=str,
        help="examples where the output of this func given a bias is true are used to train a model"
    )
    parser.add_argument(
        "--bias_2_same_as",
        default=None,
        type=str,
        help="examples where biases are same as other biases"
    )
    parser.add_argument(
        "--bias_2_not_equal",
        default=None,
        type=str,
        help="examples where biases are not equal to other biases"
    )
    parser.add_argument(
        "--bias_2_top_k",
        type=int,
        default=None,
        help="examples where biases are top k",
    )
    parser.add_argument(
        "--bias_3",
        default=None,
        type=str,
        help="Specify bias type for biased training"
    )
    parser.add_argument(
        "--bias_3_larger_than",
        type=float,
        default=None,
        help="examples where biases are greater than the threshold are used to train a model",
    )
    parser.add_argument(
        "--bias_3_smaller_than",
        type=float,
        default=None,
        help="examples where biases are smaller than the threshold are used to train a model",
    )
    parser.add_argument(
        "--bias_3_included_in",
        default=None,
        nargs='*',
        type=str,
        help="examples where biases are included in this list are used to train a model"
    )
    parser.add_argument(
        "--bias_3_custom_func",
        default=None,
        type=str,
        help="examples where the output of this func given a bias is true are used to train a model"
    )
    parser.add_argument(
        "--bias_3_same_as",
        default=None,
        type=str,
        help="examples where biases are same as other biases"
    )
    parser.add_argument(
        "--bias_3_not_equal",
        default=None,
        type=str,
        help="examples where biases are not equal to other biases"
    )
    parser.add_argument(
        "--bias_3_top_k",
        type=int,
        default=None,
        help="examples where biases are top k",
    )
    parser.add_argument(
        "--do_fewshot_train", action="store_true", help=""
    )
    parser.add_argument(
        "--do_fewshot_unique_features", action="store_true", help=""
    )
    parser.add_argument(
        "--num_fewshot_examples",
        type=int,
        default=1024,
        help="",
    )
    parser.add_argument(
        "--num_total_examples",
        type=int,
        default=5000,
        help="",
    )
    parser.add_argument(
        "--do_blend_anti_biased", action="store_true", help=""
    )
    parser.add_argument(
        "--anti_biased_ratio",
        type=float,
        default=0.0,
        help="",
    )
    parser.add_argument(
        "--anti_bias_1",
        default=None,
        type=str,
        help="Specify bias type for anti_biased training"
    )
    parser.add_argument(
        "--anti_bias_1_larger_than",
        type=float,
        default=None,
        help="examples where anti_biases are greater than or equal to the threshold are used to train a model",
    )
    parser.add_argument(
        "--anti_bias_1_smaller_than",
        type=float,
        default=None,
        help="examples where anti_biases are smaller than or equal to the threshold are used to train a model",
    )
    parser.add_argument(
        "--anti_bias_1_included_in",
        default=None,
        nargs='*',
        type=str,
        help="examples where anti_biases are included in this list are used to train a model"
    )
    parser.add_argument(
        "--anti_bias_1_custom_func",
        default=None,
        type=str,
        help="examples where the output of this func given a anti_bias is true are used to train a model"
    )
    parser.add_argument(
        "--anti_bias_1_same_as",
        default=None,
        type=str,
        help="examples where anti_biases are same as other anti_biases"
    )
    parser.add_argument(
        "--anti_bias_1_not_equal",
        default=None,
        type=str,
        help="examples where anti_biases are not equal to other anti_biases"
    )
    parser.add_argument(
        "--anti_bias_1_top_k",
        type=int,
        default=None,
        help="examples where biases are top k",
    )
    parser.add_argument(
        "--anti_bias_2",
        default=None,
        type=str,
        help="Specify anti_bias type for anti_biased training"
    )
    parser.add_argument(
        "--anti_bias_2_larger_than",
        type=float,
        default=None,
        help="examples where anti_biases are greater than the threshold are used to train a model",
    )
    parser.add_argument(
        "--anti_bias_2_smaller_than",
        type=float,
        default=None,
        help="examples where anti_biases are smaller than the threshold are used to train a model",
    )
    parser.add_argument(
        "--anti_bias_2_included_in",
        default=None,
        nargs='*',
        type=str,
        help="examples where anti_biases are included in this list are used to train a model"
    )
    parser.add_argument(
        "--anti_bias_2_custom_func",
        default=None,
        type=str,
        help="examples where the output of this func given a anti_bias is true are used to train a model"
    )
    parser.add_argument(
        "--anti_bias_2_same_as",
        default=None,
        type=str,
        help="examples where anti_biases are same as other anti_biases"
    )
    parser.add_argument(
        "--anti_bias_2_not_equal",
        default=None,
        type=str,
        help="examples where anti_biases are not equal to other anti_biases"
    )
    parser.add_argument(
        "--anti_bias_2_top_k",
        type=int,
        default=None,
        help="examples where biases are top k",
    )
    parser.add_argument(
        "--anti_bias_3",
        default=None,
        type=str,
        help="Specify anti_bias type for anti_biased training"
    )
    parser.add_argument(
        "--anti_bias_3_larger_than",
        type=float,
        default=None,
        help="examples where anti_biases are greater than the threshold are used to train a model",
    )
    parser.add_argument(
        "--anti_bias_3_smaller_than",
        type=float,
        default=None,
        help="examples where anti_biases are smaller than the threshold are used to train a model",
    )
    parser.add_argument(
        "--anti_bias_3_included_in",
        default=None,
        nargs='*',
        type=str,
        help="examples where anti_biases are included in this list are used to train a model"
    )
    parser.add_argument(
        "--anti_bias_3_custom_func",
        default=None,
        type=str,
        help="examples where the output of this func given a anti_bias is true are used to train a model"
    )
    parser.add_argument(
        "--anti_bias_3_same_as",
        default=None,
        type=str,
        help="examples where anti_biases are same as other anti_biases"
    )
    parser.add_argument(
        "--anti_bias_3_not_equal",
        default=None,
        type=str,
        help="examples where anti_biases are not equal to other biases"
    )
    parser.add_argument(
        "--anti_bias_3_top_k",
        type=int,
        default=None,
        help="examples where biases are top k",
    )
    parser.add_argument(
        "--do_exclude_long_context", action="store_true", help=""
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--per_gpu_probe_batch_size", default=32, type=int, help="Batch size per GPU/CPU for probing."
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=2.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--num_pretrain_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--max_pretrain_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup ratio")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_pretrain_steps", type=int, default=1500, help="Log every X updates steps.")
    parser.add_argument("--logging_train_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--log_before_train", action="store_true", help="")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--project", type=str, default="", help="project name for wandb logging")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument("--debug", action="store_true", help="")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # Preprocessing
    if args.do_only_preprocess:
        logger.info("You are preprocessing files for TRAINING/PRETRAINING.")
        dataset, examples, features = load_and_cache_examples(args, tokenizer, 'train', args.preprocess_file, output_examples=True)
        logger.info(f"Num of examples: {len(examples)}")
        logger.info(f"Num of features: {len(features)}")
        sys.exit()

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if args.do_train:
        if args.local_rank in [-1, 0]:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
                logger.info(f" make directory, named {args.output_dir}")
            if not os.path.exists('./exp'):
                os.mkdir('./exp')
                logger.info(f" make directory, named ./exp")
        global_step = 0

    if (args.do_train or args.do_eval) and args.local_rank in [-1, 0]:
        wandb.init(
            project=args.project,
            name=Path(args.output_dir).name,
            config={
                "epochs": args.num_train_epochs,
                "train_batch_size": args.per_gpu_train_batch_size,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "train_dataset": Path(args.train_file).stem,
                "eval_dataset": Path(args.predict_file).stem,
                "model_type": args.model_type.lower(),
            }
        )

    # Training
    if args.do_train:
        train_dataset, train_examples, train_features = load_and_cache_examples(args, tokenizer, 'train', args.train_file, output_examples=True)

        if args.do_biased_train or args.do_fewshot_train or args.do_exclude_long_context:
            bool_indices = torch.ones(len(train_features))
            train_ids = [feat.qas_id for feat in train_features]
            data_name = Path(args.train_file).stem

        if args.do_biased_train:
            logger.info(f" Current size of training set: {bool_indices.sum().item()}")

            if args.bias_1 is not None:
                bool_indices = get_bool_of_biased_dataset(
                    data_name,
                    args.bias_1,
                    args.bias_1_larger_than,
                    args.bias_1_smaller_than,
                    args.bias_1_included_in,
                    args.bias_1_custom_func,
                    args.bias_1_same_as,
                    args.bias_1_not_equal,
                    args.bias_1_top_k,
                    train_ids,
                    bool_indices)
            if args.bias_2 is not None:
                bool_indices = get_bool_of_biased_dataset(
                    data_name,
                    args.bias_2,
                    args.bias_2_larger_than,
                    args.bias_2_smaller_than,
                    args.bias_2_included_in,
                    args.bias_2_custom_func,
                    args.bias_2_same_as,
                    args.bias_2_not_equal,
                    args.bias_2_top_k,
                    train_ids,
                    bool_indices)
            if args.bias_3 is not None:
                bool_indices = get_bool_of_biased_dataset(
                    data_name,
                    args.bias_3,
                    args.bias_3_larger_than,
                    args.bias_3_smaller_than,
                    args.bias_3_included_in,
                    args.bias_3_custom_func,
                    args.bias_3_same_as,
                    args.bias_3_not_equal,
                    args.bias_3_top_k,
                    train_ids,
                    bool_indices)

        if args.do_exclude_long_context:
            short_contex_ids = [_id for _id, num in Counter(train_ids).items() if num == 1]
            bool_indices *= torch.tensor([_id in short_contex_ids for _id in train_ids])
            logger.info(f" Current size of training set after excluding long contexts: {bool_indices.sum().item()}")

        if args.do_fewshot_train:
            if args.do_fewshot_unique_features:
                indices = torch.nonzero(bool_indices, as_tuple=False).view(-1)
                fewshot_idx = random.sample(indices.numpy().tolist(), args.num_fewshot_examples)
                bool_indices *= torch.tensor([i in fewshot_idx for i in range(len(train_features))])
            else:
                # fewshot unique features
                train_qas_id = list(set([feat.qas_id for feat, b in zip(train_features, bool_indices) if b.item() == 1]))
                n_total_train = len(train_qas_id)
                logger.info(f"The total number of unique examples: {n_total_train}")
                fewshot_qas_id = random.sample(train_qas_id, args.num_fewshot_examples)
                bool_indices *= torch.tensor([feat.qas_id in fewshot_qas_id for feat in train_features])

        if args.do_blend_anti_biased and args.do_biased_train:
            # Blending anti biased examples is only conducted when args.do_biased_train is True
            # ratio of anti biased to biased
            if args.do_fewshot_train:
                num_anti_biased = math.ceil(bool_indices.sum().item() * args.anti_biased_ratio / (1 - args.anti_biased_ratio))
            else:
                logger.info(f"The total number of training examples should be: {args.num_total_examples}")
                num_anti_biased = math.ceil(args.anti_biased_ratio * args.num_total_examples)
                num_biased = args.num_total_examples - num_anti_biased
                logger.info(f"The number of biased examples: {num_biased}")
                logger.info(f"The number of anti-biased examples: {num_anti_biased}")

                # fewshot unique examples
                train_qas_id = list(set([feat.qas_id for feat, b in zip(train_features, bool_indices) if b.item() == 1]))
                n_total_train = len(train_qas_id)
                logger.info(f"The total number of unique examples: {n_total_train}")
                biased_qas_id = random.sample(train_qas_id, num_biased)
                bool_indices *= torch.tensor([feat.qas_id in biased_qas_id for feat in train_features])
                logger.info(f" The size of the biased training set: {bool_indices.sum().item()}")

            anti_bool_indices = torch.ones(len(train_features))
            logger.info(f" Current size of anti biased training set: {anti_bool_indices.sum().item()}")

            if args.anti_bias_1 is not None:
                anti_bool_indices = get_bool_of_biased_dataset(
                    data_name,
                    args.anti_bias_1,
                    args.anti_bias_1_larger_than,
                    args.anti_bias_1_smaller_than,
                    args.anti_bias_1_included_in,
                    args.anti_bias_1_custom_func,
                    args.anti_bias_1_same_as,
                    args.anti_bias_1_not_equal,
                    args.anti_bias_1_top_k,
                    train_ids,
                    anti_bool_indices)
            if args.anti_bias_2 is not None:
                anti_bool_indices = get_bool_of_biased_dataset(
                    data_name,
                    args.anti_bias_2,
                    args.anti_bias_2_larger_than,
                    args.anti_bias_2_smaller_than,
                    args.anti_bias_2_included_in,
                    args.anti_bias_2_custom_func,
                    args.anti_bias_2_same_as,
                    args.anti_bias_2_not_equal,
                    args.anti_bias_2_top_k,
                    train_ids,
                    anti_bool_indices)
            if args.anti_bias_3 is not None:
                anti_bool_indices = get_bool_of_biased_dataset(
                    data_name,
                    args.anti_bias_3,
                    args.anti_bias_3_larger_than,
                    args.anti_bias_3_smaller_than,
                    args.anti_bias_3_included_in,
                    args.anti_bias_3_custom_func,
                    args.anti_bias_3_same_as,
                    args.anti_bias_3_not_equal,
                    args.anti_bias_3_top_k,
                    train_ids,
                    anti_bool_indices)

            train_qas_id = list(set([feat.qas_id for feat, b in zip(train_features, anti_bool_indices) if b.item() == 1]))
            n_total_train = len(train_qas_id)
            logger.info(f"The total number of unique anti biased examples: {n_total_train}")
            anti_biased_qas_id = random.sample(train_qas_id, num_anti_biased)
            anti_bool_indices *= torch.tensor([feat.qas_id in anti_biased_qas_id for feat in train_features])
            logger.info(f" The size of the anti biased training set: {anti_bool_indices.sum().item()}")
            bool_indices += anti_bool_indices

        if args.do_biased_train or args.do_fewshot_train or args.do_exclude_long_context:
            indices = torch.nonzero(bool_indices, as_tuple=False).view(-1)
            train_dataset = Subset(train_dataset, indices)
            logger.info(f" The size of the resulting training set: {len(train_dataset)}")

        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

        if args.do_online_code:
            # Added here for reproducibility
            set_seed(args)
            shuffled_dataset = Subset(
                train_dataset, indices=np.random.permutation(len(train_dataset)))
            fractions = [0.001,0.002,0.004,0.008,0.016,0.032,0.0625,0.125,0.25,0.5,1]
            val_fraction = 0.01
            total_len = len(shuffled_dataset)
            logger.info(f"Total number of examples for online code: {total_len}")
            max_num_epochs = 10
            max_train_steps = 500
            patience = -1
            use_last = False
            global_step = 0
            max_loss = - math.log(1 / args.max_seq_length) # loss of uniform distribution
            eval_loss = max_loss
            nlls = []
            nlls.append(eval_loss)
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

            train_portions = []
            eval_portions = []
            block_sizes = [math.ceil(fractions[0] * total_len)]
            for i in range(len(fractions) - 1):
                train_portions.append(Subset(shuffled_dataset,
                                      range(0, math.ceil(fractions[i] * total_len))))
                eval_portions.append(Subset(shuffled_dataset,
                                     range(math.ceil(fractions[i] * total_len), math.ceil(fractions[i + 1] * total_len))))
                block_sizes.append(math.ceil(fractions[i + 1] * total_len) - math.ceil(fractions[i] * total_len))
            for i in range(len(train_portions)):
                dev_dataset_i = Subset(train_portions[i], range(0, math.ceil(len(train_portions[i]) * val_fraction)))
                dev_sampler = SequentialSampler(dev_dataset_i)
                dev_dataloader = DataLoader(dev_dataset_i, sampler=dev_sampler, batch_size=args.eval_batch_size)
                train_dataset_i = Subset(train_portions[i], range(math.ceil(len(train_portions[i]) * val_fraction), len(train_portions[i])))
                train_sampler = RandomSampler(train_dataset_i)
                train_dataloader = DataLoader(train_dataset_i, sampler=train_sampler, batch_size=args.train_batch_size)
                eval_sampler = SequentialSampler(eval_portions[i])
                eval_dataloader = DataLoader(eval_portions[i], sampler=eval_sampler, batch_size=args.eval_batch_size)

                t_total = len(train_dataloader) // args.gradient_accumulation_steps * max_num_epochs
                t_total = max(t_total, max_train_steps)
                num_epochs = t_total // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
                logging_online_steps = 20
                min_dev_loss = dev_loss = max_loss
                global_step = 0
                early_stop = False
                early_stop = False
                num_worse_step = 0

                # re-initializaion
                del model
                model = model_class.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir if args.cache_dir else None,
                )
                model.to(args.device)
                model.zero_grad()
                train_iterator = trange(
                    0, int(num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
                )

                # Prepare optimizer and schedule (linear warmup and decay)
                optimizer, scheduler = prepare_optimizer_and_schedule(args, model, t_total)
                best_checkpoint = copy.deepcopy(model.state_dict())

                for _ in train_iterator:
                    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
                    for step, batch in enumerate(epoch_iterator):
                        model.train()
                        batch = tuple(t.to(args.device) for t in batch)
                        inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "token_type_ids": batch[2],
                            "start_positions": batch[3],
                            "end_positions": batch[4],
                        }
                        if args.model_type.split('_')[0] in ["xlm", "roberta", "distilbert", "camembert"]:
                            del inputs["token_type_ids"]
                        outputs = model(**inputs)
                        # model outputs are always tuple in transformers (see doc)
                        loss = outputs.loss
                        if args.n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps
                        if args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            if args.fp16:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                            optimizer.step()
                            scheduler.step()  # Update learning rate schedule
                            model.zero_grad()
                            global_step += 1
                            if args.local_rank in [-1, 0]:
                                metrics = {}
                                metrics["online_code/lr"] = scheduler.get_lr()[0]
                                metrics["online_code/train_loss"] = loss
                                if global_step % logging_online_steps == 0:
                                    dev_loss = evaluate_loss(args, dev_dataloader, model)
                                    if dev_loss < min_dev_loss - 0.001:
                                        min_dev_loss = dev_loss
                                        best_checkpoint = copy.deepcopy(model.state_dict())
                                    else:
                                        # stop_block
                                        num_worse_step += 1

                                    if num_worse_step >= patience and patience > 0:
                                        if best_checkpoint is not None:
                                            model.load_state_dict(best_checkpoint)
                                        else:
                                            logger.warning("best_checkpoint is None")
                                        eval_loss = evaluate_loss(args, eval_dataloader, model)
                                        nlls.append(eval_loss)
                                        early_stop = True
                                metrics["online_code/dev_loss"] = dev_loss
                                metrics["online_code/eval_loss"] = eval_loss
                                wandb.log(metrics)
                        if global_step >= t_total or early_stop:
                            epoch_iterator.close()
                            break
                    if global_step >= t_total or early_stop:
                        train_iterator.close()
                        break
                if not early_stop:
                    if best_checkpoint is not None and (not use_last):
                        model.load_state_dict(best_checkpoint)
                    else:
                        logger.warning("best_checkpoint is None")
                    eval_loss = evaluate_loss(args, eval_dataloader, model)
                    nlls.append(eval_loss)
            codelengths = [nll / math.log(2) for nll in nlls]
            if len(codelengths) != len(block_sizes):
                logger.warn("len(codelengths) != len(block_sizes)")
            mdl = sum([block_size * per_sample_codelength for block_size, per_sample_codelength in zip(block_sizes, codelengths)])
            wandb.log({"online_code/mdl": mdl})
            output_online_code = {
                "MDL": mdl,
                "nll": nlls,
                "block_size": block_sizes,
                "codelength": codelengths,
            }
            with open(Path(args.output_dir) / "online_code.pkl", "wb") as f:
                pickle.dump(output_online_code, f)
            save_json(output_online_code, Path(args.output_dir) / "online_code.json")
            logger.info(f"Saved result files {str(Path(args.output_dir) / 'online_code.json')}")
        else:
            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
            max_steps = args.max_train_steps
            num_epochs = args.num_train_epochs
            if max_steps > 0:
                t_total = max_steps
                num_epochs = max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
            else:
                t_total = len(train_dataloader) // args.gradient_accumulation_steps * num_epochs

            optimizer, scheduler = prepare_optimizer_and_schedule(args, model, t_total)

            # Train
            global_step, tr_loss = train(args, train_dataset, train_dataloader, model, tokenizer, wandb,
                optimizer, scheduler,
                t_total,
                max_steps=args.max_train_steps, num_epochs=num_epochs,
                global_step=global_step, logging_steps=args.logging_train_steps)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)  # , force_download=True)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:

        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            if 'checkpoint-' in checkpoint:
                global_step = checkpoint.split("-")[-1]
            else:
                global_step = ""
            model = model_class.from_pretrained(
                checkpoint,
                from_tf=False,
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

        logger.info("Results: {}".format(results))

        wandb.log(results)

    if args.local_rank in [-1, 0]:
        wandb.finish()

    return results


if __name__ == "__main__":
    main()
